# Result table description

## Updated five-algorithm table

Output:

```text
STMy_align/SOTA_ADD/geomorph_st_code_release/outputs/five_algorithm_mean_summary_geomorph_st.csv
```

Replacement log:

```text
STMy_align/SOTA_ADD/geomorph_st_code_release/outputs/gdeba_to_geomorph_replacement_log.csv
```

### Replacement values

| dataset | metric | original GDEBA+NR | GeoMorph-ST GDEBA+NR |
|---|---:|---:|---:|
| basic_usage_demo | PCC_mean | 0.610720 | 0.612961 |
| basic_usage_demo | CI_mean | 0.737399 | 0.753859 |
| basic_usage_demo | Pairwise_MAE_mean | 16.043410 | 6.892200 |
| basic_usage_demo | Mean_NN_Distance_mean | 13.780483 | 10.816667 |
| basic_usage_demo | Runtime_s_mean | 67.194579 | 78.572196 |
| MERFISH | PCC_mean | 0.320366 | 0.320194 |
| MERFISH | CI_mean | 0.262413 | 0.260828 |
| MERFISH | Pairwise_MAE_mean | 15.119341 | 7.674251 |
| MERFISH | Mean_NN_Distance_mean | 11.959161 | 12.134168 |
| MERFISH | Runtime_s_mean | 27.160425 | 34.361592 |

The source stage metrics are:

```text
STMy_align/SOTA_ADD/results_spateo_affine/gdeba_pipeline_structure_results/gdeba_pipeline_stage_metrics.csv
```

## Compression/shear A/B/C/D/E table

Full output:

```text
STMy_align/SOTA_ADD/geomorph_st_code_release/outputs/abcde_shear_compression_summary.csv
```

Paper-facing output:

```text
STMy_align/SOTA_ADD/geomorph_st_code_release/outputs/abcde_shear_compression_paper_table.csv
```

### Main values

| deformation | method | PCC | CI | Pairwise_MAE | Mean_NN_Distance | Runtime_s |
|---|---|---:|---:|---:|---:|---:|
| compression | A: Rigid only | 0.598662 | 0.379909 | 98.741118 | 136.844523 | 45.655133 |
| compression | B: Rigid + GDEBA | 0.603229 | 0.641456 | 20.481481 | 18.121762 | 52.009206 |
| compression | C: Rigid + GDEBA + Gate | 0.603229 | 0.641456 | 20.481481 | 18.121762 | 52.009206 |
| compression | D: Rigid + GDEBA + Gate + NR | 0.605303 | 0.675962 | 18.679231 | 16.237044 | 65.470958 |
| compression | E: Rigid + NR (Original Spateo) | 0.599846 | 0.410803 | 82.898857 | 114.907788 | 58.640654 |
| shear | A: Rigid only | 0.575629 | 0.385877 | 78.335764 | 55.527933 | 44.837073 |
| shear | B: Rigid + GDEBA | 0.600178 | 0.625207 | 20.326200 | 17.034453 | 50.775578 |
| shear | C: Rigid + GDEBA + Gate | 0.600178 | 0.625207 | 20.326200 | 17.034453 | 50.775578 |
| shear | D: Rigid + GDEBA + Gate + NR | 0.607274 | 0.700386 | 17.208950 | 14.425305 | 64.422560 |
| shear | E: Rigid + NR (Original Spateo) | 0.584874 | 0.452530 | 48.554420 | 45.180097 | 63.648670 |

