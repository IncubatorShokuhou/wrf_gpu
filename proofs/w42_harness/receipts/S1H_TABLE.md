| slot | pool | proxy seg2-3 | seg2+ median | whole (session launch) | whole (t0) | first seg | JAX peak GB | compiled_new | sweep watch | CLI wait | PSI s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P_N1 | c_auto | 8.771 | 8.771 | 18.051 | 16.94 | 4.97 | 2.0 | 0 | True | 0.0018 | 0.452526 |
| P_N2 | c_auto | 3.953 | 3.953 | 13.325 | 12.408 | 5.369 | 2.0 | 0 | True | 0.0037 | 0.425701 |
| P_O1 | demand | 5.837 | 5.837 | 16.014 | 13.449 | 5.079 | 2.0 | 0 | True | 0.0028 | 0.435427 |
| P_O2 | demand | 4.742 | 4.742 | 14.492 | 12.317 | 5.27 | 2.0 | 0 | True | 0.003 | 0.552247 |
| S_N1 | c_auto | 3.54 | 7.173 | 9.658 | 9.47 | 6.01 | 2.0 | 0 | True | 0.0041 | 1.876993 |
| S_N2 | c_auto | 3.933 | 3.308 | 4.643 | 4.48 | 4.999 | 2.0 | 0 | True | 0.0018 | 1.027249 |
| S_O1 | demand | 3.48 | 3.537 | 4.853 | 4.655 | 5.223 | 2.0 | 0 | True | 0.0032 | 1.363745 |
| S_O2 | demand | 3.369 | 3.527 | 5.425 | 5.176 | 7.802 | 2.0 | 0 | True | 0.0025 | 1.22965 |
| W0 | c_auto | 4.229 | 4.229 | 13.89 | 12.948 | 5.99 | 2.0 | 0 | True | 0.0034 | 0.452606 |

| ratio | O | N | N/O | A/A % | B/B % |
|---|---|---|---|---|---|
| S:whole_s_per_fc_h:session_launch | [4.853, 5.425] | [9.658, 4.643] | 1.3914 | 11.13 | 70.13 |
| S:whole_s_per_fc_h:measured_process_t0 | [4.655, 5.176] | [9.47, 4.48] | 1.419 | 10.6 | 71.54 |
| S:seg2plus_median | [3.537, 3.527] | [7.173, 3.308] | 1.4837 | 0.28 | 73.75 |
| P:proxy_seg2_3 | [5.837, 4.742] | [8.771, 3.953] | 1.2028 | 20.7 | 75.73 |
| P:whole_s_per_fc_h:session_launch | [16.014, 14.492] | [18.051, 13.325] | 1.0285 | 9.98 | 30.12 |

identity: {"3h_all_arms_byte_identical": true, "24h_all_arms_byte_identical": true, "3h_equal_bundle_B6_prod_cli": {"P_N1": true, "P_N2": true, "P_O1": true, "P_O2": true, "W0": true}}
WN3 par3: {"cases": {"wn3_0227": {"n_wrfout": 75, "wall_s": 450.83}, "wn3_0502": {"n_wrfout": 75, "wall_s": 442.64}, "wn3_0614": {"n_wrfout": 75, "wall_s": 446.12}}, "batch_wall_s": 450.83, "s_per_case_sim_hour": 6.262, "speedup_vs_cpu_wrf_3x4core": 19.71, "consumer": "BUNDLE_CONSUMER rc=0 compiled_new=0 aliases_new=0 runtime_new=1572 other_new=0 layer_compiled=0 receipt=<DATA_ROOT>/wrf_gpu2/scratch/w42-harness/s1h/layer_wn3/runs/20261019T053405Z_2087410.json", "watch": {"sweep_rule": {"clean": true, "within_rule": true, "max_proc_seconds": 0.0, "agg_seconds": 0.0}, "strict_20_50_shadow": {"clean": false, "within_rule": true, "max_proc_seconds": 13.5, "agg_seconds": 6.8}, "cli_main_wait_share": 0.0335, "cgroup_psi_some_s": 16.072081}, "disk": {"await_ms": 19.36, "seconds": 461}}
aborted tries: []
