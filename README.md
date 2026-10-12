# GeoMorph-ST / GDEBA-Gated spatial transcriptomics slice registration

This folder is a compact, reproducible release package for the GeoMorph-ST
experiments used in the current manuscript tables.

It contains:

- a five-algorithm real-data summary with the legacy `GDEBA+NR` rows replaced
  by the verified `GeoMorph-ST GDEBA+NR` pipeline results;
- compression/shear A/B/C/D/E ablation tables for complex physical deformation;
- compression/shear visualization triptychs comparing before alignment,
  Original Spateo NR and GeoMorph-ST GDEBA+Gate+NR;
- command-line scripts, configs, lightweight metric utilities and documentation.

No raw data are copied into this release folder.  The scripts use the existing
project paths under `/ldap_shared/home/t_jyy/STAlign_project`.

## Method overview

GeoMorph-ST uses the following registration structure:

```text
Spateo rigid alignment
  -> GDEBA affine pre-registration
  -> geometry-biology consistency Gate
  -> Spateo morpho_align non-rigid refinement
```

GDEBA means geometry-dominant, expression-guided, boundary-aware affine
pre-registration.  The Gate accepts the affine initialization only when it is
physically safe, geometrically beneficial and biologically preserved.  The final
non-rigid stage reuses Spateo `morpho_align`; it is not redefined as a new
non-rigid algorithm.

## A/B/C/D/E ablation definitions

| Code | Method | Meaning |
|---|---|---|
| A | Rigid only | Spateo rigid alignment only |
| B | Rigid + GDEBA | Rigid alignment followed by raw GDEBA affine correction |
| C | Rigid + GDEBA + Gate | GDEBA output after geometry-biology Gate; no non-rigid stage |
| D | Rigid + GDEBA + Gate + NR | Full GeoMorph-ST initialization followed by Spateo NR |
| E | Original Spateo NR | Spateo rigid + Spateo non-rigid without GDEBA/Gate |

## Data used by these release tables

- `basic_usage_demo`: adjacent basic usage pair from the existing project.
- `MERFISH`: 11 adjacent slice pairs from the current MERFISH multi-slice setup.
- `compression` and `shear`: synthetic/physical deformation cases generated
  from the basic usage moving slice and aligned to `basic_usage_demo_2.h5ad`.

Original data remain in the parent project and are not modified by this release.

## Environment

Observed shell environment while packaging:

- OS: Linux `gpu6` node.
- Python seen by the packaging shell: `Python 3.13.7`.
- Main experiment scripts were originally run with the project Spateo/STAlign
  environments, especially the `stalign` environment path used by upstream
  worker scripts.
- Runtime-capable scripts expose `--device cuda:6`; if CUDA is unavailable in
  the active shell, rerun inside the same conda environment used for the
  original experiments.

Main dependencies are listed in `requirements.txt`.  SANTO, STalign and
STAligner require their own project-specific conda environments; this release
does not vendor those third-party packages.

## Reproduce release outputs

From the project root:

```bash
cd /ldap_shared/home/t_jyy/STAlign_project

# Replace legacy GDEBA+NR rows with GeoMorph-ST GDEBA+NR rows.
python STMy_align/SOTA_ADD/geomorph_st_code_release/scripts/replace_gdeba_with_geomorph_results.py

# Equivalent wrapper for the five-algorithm table.
python STMy_align/SOTA_ADD/geomorph_st_code_release/scripts/run_five_algorithm_basic_merfish.py

# Extract compression/shear A/B/C/D/E ablation table.
python STMy_align/SOTA_ADD/geomorph_st_code_release/scripts/run_abcde_shear_compression.py

# Optional: rerun upstream compression/shear ABCDE experiment on GPU6 first.
python STMy_align/SOTA_ADD/geomorph_st_code_release/scripts/run_abcde_shear_compression.py \
  --rerun-upstream --device cuda:6
```

## Outputs

- `outputs/five_algorithm_mean_summary_geomorph_st.csv`
  - five-method table with `GDEBA+NR` replaced by GeoMorph-ST GDEBA+NR.
- `outputs/gdeba_to_geomorph_replacement_log.csv`
  - exact old/new values and replacement source path.
- `outputs/abcde_shear_compression_summary.csv`
  - full compression/shear A/B/C/D/E table.
- `outputs/abcde_shear_compression_paper_table.csv`
  - compact paper-facing columns.
- `outputs/figures/compression_spateo_vs_gdeba_gate_nr_triptych.png`
  and `.pdf`
  - compression visualization:
    `Before alignment | Original Spateo NR | Rigid + GDEBA + Gate + Spateo NR`.
- `outputs/figures/shear_spateo_vs_gdeba_gate_nr_triptych.png`
  and `.pdf`
  - shear visualization:
    `Before alignment | Original Spateo NR | Rigid + GDEBA + Gate + Spateo NR`.
- `outputs/figures/figure_manifest.txt`
  - list of release figure files.

## Metrics

- PCC: expression correlation consistency, higher is better.
- CI: label/domain consistency index, higher is better.
- Pairwise MAE: nearest-neighbor pointwise spatial error, lower is better.
- Mean NN Distance: symmetric nearest-neighbor distance, lower is better.
- Runtime: end-to-end wall-clock seconds recorded by the experiment pipeline.

## Important notes

- Do not mix notebook-only Spateo `morpho_align` results with the ABCDE E-row
  unless the same input, direction, seed and evaluation protocol are used.
- `Original Spateo NR` is kept as the original rigid + non-rigid baseline.
- The replacement source for GeoMorph-ST GDEBA+NR is:

```text
STMy_align/SOTA_ADD/results_spateo_affine/gdeba_pipeline_structure_results/gdeba_pipeline_stage_metrics.csv
```

Specifically, the release uses stage `D` rows where
`stage == GeoMorph-ST GDEBA+NR`.

