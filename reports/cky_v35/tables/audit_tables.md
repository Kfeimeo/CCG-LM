### Micro-batch sweep (real data, 4096 CE tokens per update)

| micro_batch_size | status | update_wall_median_ms | update_wall_min_ms | micro_batches_per_update_median | micro_batch_wall_median_ms | ce_tokens_per_second | peak_allocated_gib | mean_padded_words | mean_word_padding_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 67075.58 | 64490.72 | 346 | 195.43 | 61.12 | 2.72 | 13.46 | 0.27 |
| 2 | ok | 35853.75 | 35341.06 | 173 | 209.02 | 114.46 | 2.73 | 16.16 | 0.41 |
| 4 | ok | 23816.39 | 19581.45 | 87 | 273.75 | 182.65 | 2.76 | 18.54 | 0.50 |
| 8 | ok | 16900.69 | 15914.23 | 44 | 384.11 | 187.74 | 3.17 | 20.36 | 0.54 |

### Stage envelopes per update (CUDA event envelopes; module hooks nest inside forward)

| micro_batch_size | update_wall_median_ms | forward_cuda_ms_per_update | module/syntax_cuda_ms_per_update | module/forest_cuda_ms_per_update | module/decoder_cuda_ms_per_update | module/reader_cuda_ms_per_update | backward_cuda_ms_per_update | clip_cuda_ms_per_update | optimizer_cuda_ms_per_update | sampling_collation_ms_per_update | h2d_host_ms_per_update |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 67075.58 | 33750.33 | 2533.77 | 19848.85 | 10940.88 | 4100.13 | 32478.91 | 6.55 | 31.11 | 129.23 | 116.43 |
| 2.00 | 35853.75 | 18334.01 | 1253.29 | 11523.53 | 5342.58 | 2052.75 | 17039.27 | 7.29 | 31.24 | 92.81 | 55.31 |
| 4.00 | 23816.39 | 12187.16 | 725.49 | 8036.40 | 3167.78 | 1239.06 | 11179.88 | 7.60 | 36.92 | 83.26 | 34.15 |
| 8.00 | 16900.69 | 8402.08 | 483.95 | 6149.15 | 1706.00 | 708.11 | 8401.91 | 17.38 | 82.09 | 83.47 | 19.55 |

### Stage share of update wall (%)

| micro_batch_size | forward_% | module/syntax_% | module/forest_% | module/decoder_% | module/reader_% | backward_% | clip_% | optimizer_% | sampling_collation_% | h2d_% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 50.3 | 3.8 | 29.6 | 16.3 | 6.1 | 48.4 | 0.0 | 0.0 | 0.2 | 0.2 |
| 2.0 | 51.1 | 3.5 | 32.1 | 14.9 | 5.7 | 47.5 | 0.0 | 0.1 | 0.3 | 0.2 |
| 4.0 | 51.2 | 3.0 | 33.7 | 13.3 | 5.2 | 46.9 | 0.0 | 0.2 | 0.3 | 0.1 |
| 8.0 | 49.7 | 2.9 | 36.4 | 10.1 | 4.2 | 49.7 | 0.1 | 0.5 | 0.5 | 0.1 |

### Index-cache behaviour during the timed sweep updates (LRU 8, old builder)

| micro_batch_size | status | timed_updates | timed_hits | timed_misses | timed_hit_rate | build_ms_per_update | build_share_of_update |
|---|---|---|---|---|---|---|---|
| 1 | ok | 3 | 444 | 932 | 0.323 | 3645.230 | 0.054 |
| 2 | ok | 3 | 337 | 351 | 0.490 | 1872.898 | 0.052 |
| 4 | ok | 3 | 229 | 117 | 0.662 | 948.488 | 0.042 |
| 8 | ok | 5 | 244 | 63 | 0.795 | 1150.274 | 0.053 |

### Real key stream (100 updates): unique keys, LRU hit rates by capacity

| micro_batch_size | micro_batches_per_update | unique_cache_keys | lru8_hit_rate | lru16_hit_rate | lru32_hit_rate | lru64_hit_rate | lru128_hit_rate |
|---|---|---|---|---|---|---|---|
| 1.000 | 344.730 | 84.000 | 0.336 | 0.612 | 0.881 | 0.987 | 0.998 |
| 2.000 | 172.600 | 79.000 | 0.510 | 0.867 | 0.970 | 0.994 | 0.995 |
| 4.000 | 86.550 | 57.000 | 0.652 | 0.983 | 0.991 | 0.993 | 0.993 |
| 8.000 | 43.570 | 64.000 | 0.780 | 0.975 | 0.982 | 0.985 | 0.985 |
| 16.000 | 22.030 | 77.000 | 0.885 | 0.954 | 0.960 | 0.964 | 0.965 |
| 32.000 | 11.200 | 77.000 | 0.879 | 0.908 | 0.921 | 0.930 | 0.931 |
| 64.000 | 6.000 | 72.000 | 0.807 | 0.838 | 0.857 | 0.880 | 0.880 |
| 128.000 | 3.000 | 37.000 | 0.730 | 0.780 | 0.873 | 0.877 | 0.877 |

### Padding / utilization of the real stream

| micro_batch_size | padded_words_mean | history_slots_mean | target_length_padded_mean | word_padding_ratio_mean | span_utilization_mean | target_padding_ratio_mean | duplicate_history_fraction |
|---|---|---|---|---|---|---|---|
| 1.000 | 13.355 | 3.526 | 11.898 | 0.274 | 0.615 | 0.001 | 0.000 |
| 2.000 | 16.169 | 3.918 | 14.599 | 0.424 | 0.438 | 0.172 | 0.000 |
| 4.000 | 18.428 | 3.996 | 17.206 | 0.499 | 0.345 | 0.292 | 0.001 |
| 8.000 | 20.383 | 3.999 | 19.661 | 0.547 | 0.286 | 0.378 | 0.001 |
| 16.000 | 22.004 | 3.997 | 22.143 | 0.582 | 0.245 | 0.446 | 0.001 |
| 32.000 | 23.089 | 3.995 | 24.692 | 0.603 | 0.222 | 0.500 | 0.003 |
| 64.000 | 23.677 | 4.000 | 27.453 | 0.613 | 0.210 | 0.548 | 0.004 |
| 128.000 | 23.983 | 4.000 | 30.813 | 0.620 | 0.203 | 0.597 | 0.007 |

### Micro-batch decomposition: wall vs GPU busy (profiled run) and launches

| micro_batch_size | batch_index | ce_tokens | padded_words | plain_wall_median_ms | plain_cuda_envelope_median_ms | ce_tokens_per_second_microbatch | profiled_wall_ms | profiled_gpu_busy_ms | profiled_gpu_idle_fraction | cuda_launches | graph_score_prior_autograd_nodes | graph_loss_autograd_nodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 30.00 | 18.00 | 322.90 | 322.77 | 92.91 | 579.37 | 68.03 | 0.88 | 5000.00 | 679.00 | 2346.00 |
| 4.00 | 1.00 | 35.00 | 21.00 | 352.71 | 352.59 | 99.23 | 489.97 | 60.80 | 0.88 | 5575.00 | 781.00 | 2500.00 |
| 4.00 | 2.00 | 30.00 | 18.00 | 334.88 | 334.77 | 89.58 | 489.08 | 58.84 | 0.88 | 4980.00 | 679.00 | 2346.00 |
| 8.00 | 0.00 | 65.00 | 21.00 | 373.81 | 373.58 | 173.88 | 503.26 | 69.35 | 0.86 | 5708.00 | 781.00 | 2526.00 |
| 8.00 | 1.00 | 79.00 | 18.00 | 326.61 | 326.48 | 241.88 | 438.93 | 62.21 | 0.86 | 5026.00 | 679.00 | 2346.00 |
| 8.00 | 2.00 | 75.00 | 18.00 | 295.63 | 295.52 | 253.69 | 440.91 | 59.49 | 0.86 | 4915.00 | 679.00 | 2320.00 |
| 16.00 | 0.00 | 144.00 | 21.00 | 323.93 | 323.83 | 444.54 | 480.83 | 74.40 | 0.85 | 5675.00 | 781.00 | 2526.00 |
| 16.00 | 1.00 | 169.00 | 24.00 | 370.76 | 370.64 | 455.82 | 517.83 | 81.58 | 0.84 | 6038.00 | 892.00 | 2663.00 |
| 16.00 | 2.00 | 168.00 | 21.00 | 347.50 | 347.36 | 483.45 | 497.68 | 79.48 | 0.84 | 5582.00 | 781.00 | 2526.00 |

### Module/stage CUDA envelopes per micro-batch (hooked run, no internal sync)

| micro_batch_size | batch_index | backward_cuda_ms | clip_cuda_ms | forest/search_cuda_ms | forward_cuda_ms | h2d_cuda_ms | module/composer_cuda_ms | module/decoder_cuda_ms | module/forest_cuda_ms | module/reader_cuda_ms | module/span_scorer_cuda_ms | module/syntax_cuda_ms | optimizer_cuda_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 130.46 | 7.45 | 56.78 | 148.36 | 0.20 | 18.18 | 39.54 | 96.39 | 14.45 | 1.49 | 9.42 | 33.20 |
| 4.00 | 1.00 | 158.86 | 6.67 | 69.36 | 157.30 | 0.32 | 17.52 | 41.38 | 102.99 | 16.68 | 1.71 | 10.12 | 32.02 |
| 4.00 | 2.00 | 144.09 | 7.62 | 70.94 | 159.99 | 0.36 | 15.62 | 41.87 | 105.09 | 15.42 | 1.48 | 9.14 | 32.88 |
| 8.00 | 0.00 | 157.80 | 7.60 | 70.87 | 163.69 | 0.26 | 19.07 | 40.34 | 112.63 | 15.33 | 1.65 | 10.14 | 31.12 |
| 8.00 | 1.00 | 130.72 | 9.61 | 58.90 | 146.37 | 0.30 | 17.25 | 38.13 | 97.17 | 13.91 | 1.69 | 9.43 | 30.93 |
| 8.00 | 2.00 | 127.56 | 6.56 | 57.30 | 133.81 | 0.23 | 12.83 | 35.46 | 90.89 | 13.07 | 1.52 | 9.50 | 29.08 |
| 16.00 | 0.00 | 129.46 | 7.00 | 64.97 | 167.95 | 0.28 | 18.05 | 43.32 | 112.40 | 16.94 | 2.56 | 9.17 | 29.20 |
| 16.00 | 1.00 | 164.52 | 7.45 | 90.48 | 188.85 | 0.28 | 23.31 | 33.99 | 144.06 | 12.42 | 3.19 | 8.98 | 29.17 |
| 16.00 | 2.00 | 131.51 | 6.91 | 68.29 | 153.35 | 0.31 | 17.70 | 33.01 | 109.73 | 12.39 | 2.66 | 9.82 | 29.20 |

### Forest internals (collect_timing run) and D2H wait/copy split

| micro_batch_size | batch_index | forest_score_search_host_ms | forest_score_search_cuda_ms | forest_topology_d2h_blocking_ms | forest_dag_cpu_ms | forest_index_h2d_host_ms | forest_index_h2d_cuda_ms | forest_semantic_host_ms | forest_semantic_cuda_ms | d2h_gpu_wait_ms_when_synced | d2h_copy_tolist_ms_when_synced | forest_v25_actual_composer_rows | forest_v25_composer_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 66.19 | 66.72 | 0.97 | 1.86 | 0.42 | 0.49 | 25.53 | 25.53 | 0.02 | 0.82 | 374.00 | 17.00 |
| 4.00 | 1.00 | 83.90 | 83.90 | 0.98 | 2.21 | 0.88 | 1.33 | 25.54 | 25.17 | 0.06 | 1.57 | 400.00 | 19.00 |
| 4.00 | 2.00 | 67.88 | 67.88 | 1.11 | 1.23 | 0.31 | 0.36 | 23.72 | 23.72 | 0.03 | 1.00 | 241.00 | 17.00 |
| 8.00 | 0.00 | 89.36 | 89.10 | 2.18 | 3.20 | 0.51 | 0.57 | 46.20 | 46.19 | 0.04 | 2.31 | 745.00 | 20.00 |
| 8.00 | 1.00 | 71.44 | 71.53 | 1.51 | 3.07 | 0.55 | 0.63 | 26.81 | 26.81 | 0.04 | 1.46 | 513.00 | 17.00 |
| 8.00 | 2.00 | 72.67 | 72.61 | 1.95 | 3.77 | 0.43 | 0.46 | 25.93 | 25.92 | 0.04 | 1.96 | 564.00 | 16.00 |
| 16.00 | 0.00 | 132.85 | 131.59 | 3.77 | 9.55 | 0.77 | 0.82 | 34.89 | 34.88 | 0.03 | 3.03 | 1307.00 | 20.00 |
| 16.00 | 1.00 | 90.52 | 88.78 | 3.67 | 5.50 | 0.50 | 0.52 | 25.86 | 25.86 | 0.02 | 3.38 | 1171.00 | 21.00 |
| 16.00 | 2.00 | 84.11 | 82.85 | 3.15 | 5.46 | 0.53 | 0.57 | 32.20 | 32.19 | 0.03 | 2.81 | 1237.00 | 20.00 |

### Kernel-active time attributed to ranges (profiled, ms)

| micro_batch_size | batch_index | kernel_active_step_ms | kernel_active_phase/h2d_ms | kernel_active_phase/forward_ms | kernel_active_module/syntax_ms | kernel_active_module/forest_ms | kernel_active_forest/search_ms | kernel_active_module/composer_ms | kernel_active_module/decoder_ms | kernel_active_module/reader_ms | kernel_active_phase/backward_ms | kernel_active_phase/clip_ms | kernel_active_phase/optimizer_ms | kernel_active_unattributed_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 0.00 | 0.00 | 0.06 | 1.02 | 0.48 | 0.10 | 0.86 | 3.65 | 0.72 | 13.53 | 4.26 | 40.44 | 0.00 |
| 4.00 | 1.00 | 0.00 | 0.00 | 0.06 | 1.10 | 0.45 | 0.10 | 0.96 | 3.62 | 0.69 | 13.73 | 4.37 | 32.22 | 0.00 |
| 4.00 | 2.00 | 0.00 | 0.00 | 0.05 | 1.01 | 0.47 | 0.10 | 0.76 | 3.61 | 0.70 | 12.59 | 4.30 | 32.38 | 0.00 |
| 8.00 | 0.00 | 0.00 | 0.00 | 0.10 | 1.97 | 0.56 | 0.10 | 1.08 | 4.08 | 0.68 | 17.16 | 4.29 | 35.53 | 0.00 |
| 8.00 | 1.00 | 0.00 | 0.00 | 0.10 | 1.61 | 0.46 | 0.09 | 0.87 | 4.11 | 0.70 | 16.02 | 4.31 | 30.89 | 0.00 |
| 8.00 | 2.00 | 0.00 | 0.00 | 0.10 | 1.62 | 0.44 | 0.09 | 0.82 | 4.11 | 0.69 | 15.28 | 4.40 | 28.94 | 0.00 |
| 16.00 | 0.00 | 0.00 | 0.00 | 0.18 | 3.50 | 0.77 | 0.10 | 1.18 | 5.56 | 0.83 | 24.58 | 4.37 | 29.06 | 0.00 |
| 16.00 | 1.00 | 0.00 | 0.00 | 0.20 | 4.25 | 0.89 | 0.10 | 1.15 | 6.76 | 1.08 | 28.49 | 4.38 | 28.91 | 0.00 |
| 16.00 | 2.00 | 0.00 | 0.00 | 0.23 | 3.50 | 0.80 | 0.10 | 1.16 | 6.87 | 1.10 | 27.05 | 4.33 | 30.10 | 0.00 |

### Score-search kernel-active time and launches by sub-stage (sum over widths)

| micro_batch_size | batch_index | search_gather_kernel_ms | search_gather_launches | search_pair_kernel_ms | search_pair_launches | search_selection_kernel_ms | search_selection_launches | search_decode_kernel_ms | search_decode_launches | search_assembly_kernel_ms | search_assembly_launches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 0.68 | 253.00 | 0.28 | 171.00 | 1.48 | 522.00 | 0.12 | 34.00 | 0.34 | 167.00 |
| 4.00 | 1.00 | 0.84 | 328.00 | 0.33 | 201.00 | 1.80 | 624.00 | 0.15 | 40.00 | 0.39 | 197.00 |
| 4.00 | 2.00 | 0.68 | 253.00 | 0.28 | 171.00 | 1.48 | 522.00 | 0.12 | 34.00 | 0.33 | 167.00 |
| 8.00 | 0.00 | 0.88 | 328.00 | 0.33 | 201.00 | 2.02 | 624.00 | 0.19 | 40.00 | 0.41 | 197.00 |
| 8.00 | 1.00 | 0.66 | 253.00 | 0.28 | 171.00 | 1.60 | 522.00 | 0.15 | 34.00 | 0.34 | 167.00 |
| 8.00 | 2.00 | 0.65 | 253.00 | 0.28 | 171.00 | 1.58 | 522.00 | 0.15 | 34.00 | 0.34 | 167.00 |
| 16.00 | 0.00 | 0.89 | 328.00 | 0.34 | 201.00 | 2.41 | 624.00 | 0.21 | 40.00 | 0.42 | 197.00 |
| 16.00 | 1.00 | 1.13 | 412.00 | 0.41 | 231.00 | 3.07 | 726.00 | 0.24 | 46.00 | 0.49 | 227.00 |
| 16.00 | 2.00 | 0.87 | 328.00 | 0.34 | 201.00 | 2.41 | 624.00 | 0.21 | 40.00 | 0.42 | 197.00 |

### Staged (detached-boundary, synchronized) phase envelopes per micro-batch

| micro_batch_size | batch_index | staged_module/syntax_cuda_ms | staged_module/composer_cuda_ms | staged_module/cky_cuda_ms | staged_module/reader_cuda_ms | staged_module/decoder_cuda_ms | staged_phase/forward_cuda_ms | staged_module/reader_backward_cuda_ms | staged_phase/decoder_backward_cuda_ms | staged_module/composer_backward_cuda_ms | staged_phase/cky_backward_cuda_ms | staged_phase/syntax_embedding_backward_cuda_ms | staged_phase/clip_optimizer_cuda_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 10.10 | 26.57 | 99.41 | 15.89 | 40.04 | 151.48 | 14.01 | 47.62 | 29.25 | 67.74 | 13.03 | 40.85 |
| 4.00 | 1.00 | 10.51 | 20.56 | 117.64 | 15.08 | 39.19 | 169.08 | 17.43 | 59.67 | 37.98 | 91.15 | 16.69 | 40.37 |
| 4.00 | 2.00 | 13.03 | 22.42 | 126.47 | 19.46 | 48.71 | 190.05 | 18.97 | 64.13 | 37.44 | 91.90 | 14.35 | 43.29 |
| 8.00 | 0.00 | 9.22 | 21.93 | 121.26 | 17.44 | 46.17 | 178.04 | 16.65 | 54.28 | 33.65 | 84.10 | 12.45 | 38.49 |
| 8.00 | 1.00 | 9.72 | 18.64 | 107.80 | 18.14 | 44.90 | 163.94 | 18.86 | 63.68 | 39.51 | 89.48 | 13.43 | 42.71 |
| 8.00 | 2.00 | 8.65 | 15.52 | 87.43 | 13.62 | 36.42 | 133.92 | 15.61 | 53.01 | 28.57 | 68.58 | 12.44 | 36.25 |
| 16.00 | 0.00 | 10.24 | 23.39 | 123.62 | 13.81 | 35.79 | 171.17 | 15.53 | 54.81 | 38.80 | 90.67 | 13.83 | 37.43 |
| 16.00 | 1.00 | 10.56 | 25.75 | 147.50 | 16.13 | 44.03 | 203.50 | 18.11 | 66.44 | 33.62 | 87.48 | 12.27 | 36.34 |
| 16.00 | 2.00 | 11.45 | 18.61 | 110.26 | 13.98 | 36.26 | 159.67 | 13.69 | 49.84 | 36.24 | 88.23 | 12.59 | 35.79 |

### Staged phases: kernel-active (profiled) ms

| micro_batch_size | batch_index | staged_kernel_active_phase/forward_ms | staged_kernel_active_phase/decoder_backward_ms | staged_kernel_active_module/reader_backward_ms | staged_kernel_active_phase/cky_backward_ms | staged_kernel_active_module/composer_backward_ms | staged_kernel_active_phase/syntax_embedding_backward_ms | staged_kernel_active_phase/clip_optimizer_ms |
|---|---|---|---|---|---|---|---|---|
| 4.00 | 0.00 | 0.06 | 6.19 | 0.86 | 2.24 | 1.87 | 1.83 | 35.65 |
| 4.00 | 1.00 | 0.06 | 6.12 | 0.84 | 2.55 | 2.06 | 2.10 | 35.60 |
| 4.00 | 2.00 | 0.05 | 6.09 | 0.85 | 2.13 | 1.79 | 1.83 | 35.26 |
| 8.00 | 0.00 | 0.10 | 7.30 | 1.00 | 2.86 | 2.36 | 3.58 | 35.22 |
| 8.00 | 1.00 | 0.10 | 7.46 | 1.02 | 2.19 | 1.94 | 3.03 | 33.30 |
| 8.00 | 2.00 | 0.10 | 7.35 | 0.99 | 2.10 | 1.83 | 2.98 | 33.33 |
| 16.00 | 0.00 | 0.18 | 10.66 | 1.34 | 3.47 | 2.58 | 6.62 | 33.30 |
| 16.00 | 1.00 | 0.21 | 12.53 | 1.53 | 3.95 | 2.60 | 7.85 | 33.30 |
| 16.00 | 2.00 | 0.22 | 12.85 | 1.55 | 3.53 | 2.51 | 6.59 | 33.32 |

### Python exclusive time per micro-batch step (cProfile, ms)

| micro_batch_size | batch_index | python_project_ms | python_torch_ms | python_builtin_ms |
|---|---|---|---|---|
| 4.00 | 0.00 | 36.12 | 15.62 | 276.17 |
| 4.00 | 1.00 | 46.49 | 16.65 | 358.39 |
| 4.00 | 2.00 | 40.52 | 15.54 | 315.95 |
| 8.00 | 0.00 | 45.78 | 13.72 | 310.95 |
| 8.00 | 1.00 | 32.37 | 12.36 | 246.48 |
| 8.00 | 2.00 | 31.72 | 11.96 | 240.86 |
| 16.00 | 0.00 | 40.00 | 14.82 | 275.85 |
| 16.00 | 1.00 | 47.27 | 14.73 | 303.23 |
| 16.00 | 2.00 | 44.12 | 13.26 | 307.61 |