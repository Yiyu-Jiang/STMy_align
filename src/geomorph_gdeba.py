"""GeoMorph-ST GDEBA release definitions.

This file intentionally contains the stable method definitions used in the
release tables.  The heavy numerical implementation remains in the upstream
project scripts referenced by `scripts/run_abcde_shear_compression.py`.
"""

GEOMORPH_METHOD_NAME = "GeoMorph-ST GDEBA+NR"

PIPELINE_STEPS = [
    "Spateo rigid alignment",
    "GDEBA affine pre-registration",
    "Geometry-biology consistency Gate",
    "Spateo morpho_align non-rigid refinement",
]


def pipeline_description() -> str:
    return " -> ".join(PIPELINE_STEPS)
