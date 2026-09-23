#!/bin/bash
# Start and stop the ML container around actual pipeline work.
#
# PaddleOCR and MiniLM keep their weights resident, so the container costs
# several hundred MB for as long as it runs, while the pipeline needs it
# only in bursts. The backend cannot start a container itself without the
# Docker socket, which is root-equivalent, so it touches a sentinel file
# instead and this script -- which does have Docker access -- acts on it.
#
#   wake   started by a systemd .path unit when the sentinel changes
#   reap   started by a systemd .timer to stop the container once idle
#
# The sentinel's mtime is the only shared state.
set -euo pipefail

SENTINEL="${ALETHEIA_ML_SENTINEL:-/home/invictus/aletheia/run/ml.wanted}"
CONTAINER="${ALETHEIA_ML_CONTAINER:-aletheia-ml-1}"
IDLE_SECONDS="${ALETHEIA_ML_IDLE_SECONDS:-3600}"

running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}

case "${1:-}" in
  wake)
    if running; then
      echo "already running"
    else
      docker start "$CONTAINER"
      echo "started $CONTAINER"
    fi
    ;;

  reap)
    running || { echo "already stopped"; exit 0; }
    # No sentinel means nothing has ever asked for it; treat as idle.
    if [ -f "$SENTINEL" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$SENTINEL") ))
    else
      age="$IDLE_SECONDS"
    fi
    if [ "$age" -ge "$IDLE_SECONDS" ]; then
      docker stop "$CONTAINER" >/dev/null
      echo "stopped $CONTAINER after ${age}s idle"
    else
      echo "active ${age}s ago, keeping it up"
    fi
    ;;

  status)
    running && state=running || state=stopped
    if [ -f "$SENTINEL" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$SENTINEL") ))
      echo "$CONTAINER $state, last activity ${age}s ago, idle limit ${IDLE_SECONDS}s"
    else
      echo "$CONTAINER $state, no activity recorded"
    fi
    ;;

  *)
    echo "usage: $0 {wake|reap|status}" >&2
    exit 64
    ;;
esac
