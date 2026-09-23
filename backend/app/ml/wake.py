"""On-demand lifecycle for the ML service.

PaddleOCR and MiniLM hold their weights resident, so the ``ml`` container
costs several hundred MB of RAM for as long as it runs, and the pipeline
needs it for minutes at a time at most. A deployment can therefore keep it
stopped and start it only when work arrives.

The backend cannot start a container itself without being handed the Docker
socket, which is root-equivalent and not something a public-facing API
should hold. Instead it touches a sentinel file on a shared volume: a host
supervisor watches that path and starts the container, and a periodic check
stops it again once the file has gone stale. The file's mtime is the single
piece of shared state, which means the API needs no privileges at all and
the whole arrangement degrades to nothing when the file is not configured.

Locally there is no supervisor and the container simply runs, so every
function here is a no-op beyond a cheap ``touch``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from app.ml.client import MLServiceClient, MLServiceError

__all__ = ["mark_activity", "wake_and_wait"]

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0


def _sentinel() -> Path | None:
    """The path the host supervisor watches, or None when not deployed."""
    raw = os.environ.get("ML_WAKE_FILE", "").strip()
    return Path(raw) if raw else None


def mark_activity() -> None:
    """Record that pipeline work is happening, so the idle check holds off.

    Every stage counts, not only the ones that call the ML service:
    normalization never touches it, but a run that is still normalizing is
    not idle, and stopping the container underneath the matching stage that
    follows would be wrong.
    """
    path = _sentinel()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError as exc:
        # A missing volume means nobody is supervising; that is not an error.
        logger.debug("ML activity marker unavailable at %s: %s", path, exc)


def wake_and_wait(
    client: MLServiceClient, timeout_s: float = 90.0
) -> bool:
    """Ask for the ML service and block until it answers, or give up.

    Returns True when it is ready. A False return is not fatal anywhere:
    OCR is a fallback that ingestion survives without, and embeddings fall
    back to lexical discovery. Callers should carry on either way rather
    than failing the request.
    """
    mark_activity()

    deadline = time.monotonic() + max(0.0, timeout_s)
    attempted = False
    while True:
        try:
            client.health(timeout_s=3.0)
            if attempted:
                logger.info("ML service ready")
            return True
        except MLServiceError:
            pass
        attempted = True
        if time.monotonic() >= deadline:
            logger.warning(
                "ML service did not become ready within %.0fs; continuing "
                "without it",
                timeout_s,
            )
            return False
        time.sleep(_POLL_INTERVAL_S)
