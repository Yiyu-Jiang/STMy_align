# Experiment protocol

## Five-algorithm basic/MERFISH comparison

Input table:

```text
STMy_align/SOTA_ADD/results_spateo_affine/five_algorithm_basic_merfish_gpu/summary/five_algorithm_mean_summary.csv
```

Replacement source:

```text
STMy_align/SOTA_ADD/results_spateo_affine/gdeba_pipeline_structure_results/gdeba_pipeline_stage_metrics.csv
```

The replacement script preserves rows for:

- `Original Spateo NR`
- `SANTO`
- `STalign`
- `STAligner`

Only rows with `algorithm == GDEBA+NR` are replaced.  Replacement values are
computed from successful stage-D rows:

```text
stage_code == D
stage == GeoMorph-ST GDEBA+NR
Status == SUCCESS
```

## Compression/shear A/B/C/D/E ablation

Input table:

```text
STMy_align/SOTA_ADD/results_spateo_affine/abcde_basic_shear_compression_gpu_rerun/summary/ABCDE_basic_usage_shear_compression_gpu_metrics.csv
```

The release table keeps only:

- `compression`
- `shear`

and preserves all A/B/C/D/E columns and metrics from the recomputed GPU-rigid
run.

## Runtime protocol

For A/B/C/D/E:

- A runtime = Spateo rigid worker runtime.
- B/C runtime = rigid runtime + GDEBA affine runtime.
- D runtime = rigid runtime + GDEBA affine runtime + non-rigid runtime from the
  GDEBA/Gate initialization.
- E runtime = rigid runtime + original Spateo non-rigid runtime.

The release does not recalculate the heavy alignment unless
`--rerun-upstream` is explicitly passed.

