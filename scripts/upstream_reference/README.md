# Verified upstream implementation files

This directory contains exact copies of the project scripts used to generate
or evaluate the GeoMorph-ST release results. They are kept together because
the upstream modules use sibling imports.

Core files requested for the release:

- `gdeba_adaptive_expression.py`
- `run_gdeba_affine.py`
- `gdeba_dual_gate.py`

Required sibling dependencies:

- `run_expression_guided_affine.py`
- `compute_spateo_affine_metrics.py`
- `isbi_gate_config.json`

The three core files were copied without modifying their numerical behavior.
Their SHA-256 hashes therefore match the source files in `STMy_align/SOTA_ADD`.
The recorded hashes are available in `SOURCE_MANIFEST.sha256` and can be
verified from this directory with `sha256sum -c SOURCE_MANIFEST.sha256`.

Run the affine entry point from the release root with an environment that has
the dependencies listed in the top-level `requirements.txt`:

```bash
python scripts/upstream_reference/run_gdeba_affine.py --help
```

Raw `.h5ad` data are not vendored in this repository. Supply project-local
paths using the command-line arguments when executing a full experiment.
