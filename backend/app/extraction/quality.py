"""Page-level extraction quality scoring (Phase 1, PLAN §39).

Deterministic routing signal: GOOD means native extraction is sufficient,
BAD means the page is flagged NEEDS_OCR for future Phase 3 routing.
This module performs no OCR itself.
"""

from app.core.config import settings
from app.models.page import QualityVerdict

# Saturation points: char/word counts at which that component is "full".
_CHAR_SATURATION: float = 500.0
_WORD_SATURATION: float = 100.0
# Sparse image-bearing pages (below extraction_min_chars) are discounted.
_IMAGE_SPARSE_PENALTY: float = 0.5


def assess_quality(
    char_count: int, word_count: int, suspicious_ratio: float, has_images: bool
) -> tuple[float, QualityVerdict]:
    """Score native text extraction quality deterministically.

    Formula (all inputs clamped to valid ranges first):
        char_component  = min(max(char_count, 0) / 500.0, 1.0)
        word_component  = min(max(word_count, 0) / 100.0, 1.0)
        base            = 0.6 * char_component + 0.4 * word_component
        garble_factor   = (1.0 - clamp(suspicious_ratio, 0, 1)) ** 2
        score           = base * garble_factor
        if has_images and char_count < settings.extraction_min_chars:
            score *= 0.5
        score clamped to [0.0, 1.0]

    Rules:
        - char_count == 0 (empty page) -> (0.0, BAD), short-circuit.
        - Verdict GOOD iff score >= settings.extraction_quality_threshold,
          else BAD. BAD means "flagged NEEDS_OCR for future Phase 3 routing".
        - The quadratic garble_factor drags heavy suspicious_ratio down
          sharply (0.5 -> x0.25; 0.8 -> x0.04).
        - Image-only/sparse pages score low via the 0.5 sparse-image penalty
          on top of an already-low base density score.

    Stdlib only. No I/O, no randomness, no OCR.
    """
    if char_count == 0:
        return 0.0, QualityVerdict.BAD

    chars = max(char_count, 0)
    words = max(word_count, 0)
    suspicious = min(max(suspicious_ratio, 0.0), 1.0)

    char_component = min(chars / _CHAR_SATURATION, 1.0)
    word_component = min(words / _WORD_SATURATION, 1.0)
    base = 0.6 * char_component + 0.4 * word_component

    garble_factor = (1.0 - suspicious) ** 2
    score = base * garble_factor

    if has_images and chars < settings.extraction_min_chars:
        score *= _IMAGE_SPARSE_PENALTY

    score = min(max(score, 0.0), 1.0)

    verdict = (
        QualityVerdict.GOOD
        if score >= settings.extraction_quality_threshold
        else QualityVerdict.BAD
    )
    return score, verdict


def needs_ocr(verdict: QualityVerdict) -> bool:
    """Whether a page's verdict routes it to OCR fallback.

    Literal PLAN routing: any BAD verdict triggers OCR. GOOD pages
    never do. The caller additionally requires an available provider;
    this predicate is purely verdict-driven and document-agnostic.
    """
    return verdict == QualityVerdict.BAD
