"""Spateo baseline labels used by the release scripts."""

ORIGINAL_SPATEO_NR = "Original Spateo NR"
SPATEO_RIGID = "Spateo rigid"


def is_original_spateo_nr(method: str) -> bool:
    return str(method).strip() == ORIGINAL_SPATEO_NR
