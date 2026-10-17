# Heavy tests and CI tiers

Only paid executions; RSS is parent process HWM. Cgroup peak is separate. Unknown/lower-bound peaks are labelled. Measured cap is not a minimum. Active arms remain incomplete.

| Test | Peak RSS GiB | Measured cap | Outcome / tier |
|---|---:|---|---|
| `tests/dynamics/test_sms3dtke_full.py::test_sms_complete_compiled_pristine_operator` | 12.525 | 24G/noSwap | PASS |
| `tests/dynamics/test_sms3dtke_full.py::test_sms_three_own_steps_nine_rk_stages_pristine` | 3.947 | 12G/noSwap | PASS |
| `tests/test_m6_tier2_coupled.py::test_forecast_segment_pre_sanitize_tap_returns_one_state_per_step` | ≥12.037; current TRACE 1.012 | 12G/noSwap | RETIRED executed harness; current TRACE gate PASS under6G |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[11]` | 2.784 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[38]` | 8.858 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[40]` | 1.398 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[50]` | 1.798 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[51]` | 1.939 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[52]` | 5.244 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[53]` | 3.452 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[55]` | 1.567 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[56]` | 7.686 | 12G/noSwap | PASS |
| `tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[97]` | 1.152 | 12G/noSwap | PASS |
| `tests/test_v013_ra_sw_gsfc.py::test_gsfc_sw_fp64_precision_audit[5]` | 2.680 | 12G/noSwap | PASS |
| `tests/test_v0234_b2_bounded_terminal_gate.py::test_real_fixture_fused_vs_eager_is_within_recompile_floor` | 17.043 | 24G/noSwap | PASS |
| `tests/test_v060_pbl_acm2.py::test_v060_acm2_adapter_contract_keys` | 3.306 | 12G/noSwap | PASS |
| `tests/test_v060_pbl_acm2.py::test_v060_acm2_savepoint_parity_report` | 8.565 | 12G/noSwap | PASS |
| `tests/v025/b_column/test_kf_rk.py::test_actual_prod_rk_dispatch_d1_and_composed_keys[False]` | 13.232 | 24G/noSwap | PASS |
| `tests/v025/b_column/test_kf_rk.py::test_actual_prod_rk_dispatch_d1_and_composed_keys[True]` | 13.279 | 24G/noSwap | PASS |
| `tests/v025/b_core/test_h_diabatic_pair.py::test_pair_dispatch_follows_wrf_order` | 13.304 | 24G/noSwap | PASS |
| `tests/v025/b_core/test_root_normal_relax_blend.py::test_root_own_step_pins_spec_face_without_relax_row_blend[0]` | 13.181 | 24G/noSwap | PASS |
| `tests/v040/rootbdy/test_root_flow_extras.py::test_swiss_production_step_dispatches_extra_flow_dep[24]` | 5.129 | 12G/noSwap | PASS |
| `tests/v040/rootbdy/test_root_flow_extras.py::test_swiss_production_step_dispatches_extra_flow_dep[28]` | 5.297 | 12G/noSwap | PASS |
| `tests/v040/rootbdy/test_root_flow_extras.py::test_swiss_production_step_dispatches_extra_flow_dep[40]` | 4.975 | 12G/noSwap | PASS |
| `tests/v025/b_core/test_root_normal_relax_blend.py::test_root_own_step_pins_spec_face_without_relax_row_blend[1]` | unknown | unexecuted | UNEXECUTED; shared PROD setup likely heavy [I] |
| `tests/v025/b_core/test_root_normal_relax_blend.py::test_pre_fix_stage_config_is_rejected` | unknown | unexecuted | UNEXECUTED; shared PROD setup likely heavy [I] |
| `tests/v025/b_core/test_root_normal_relax_blend.py::test_nested_children_keep_their_boundary_config` | unknown | unexecuted | UNEXECUTED; shared PROD setup likely heavy [I] |
| `tests/v025/b_core/test_root_number_transport.py::test_root_own_step_transports_ni_nr` | unknown | unexecuted | UNEXECUTED; shared PROD setup likely heavy [I] |

The JSON lists every attempt, immutable source, exact receipt and original XML hashes. The two KF production cases peak during input loading, before tracing. The retired debug-tap execution is explicitly a current TRACE gate, not runtime fidelity coverage. No extra numerical validation was run to build this list.

## Additional session memory growers

These are cumulative session HWM readings at each test completion, not isolated-test peaks. All passed under6G; retained RSS does not prove a live-object leak.

| Test | Session HWM GiB | Retained RSS change GiB |
|---|---:|---:|
| `tests/test_u2_input_lai_init.py::test_real_loader_cold_lai_threshold_and_nonvegetated_cells[7]` | 5.048 | +3.989 |
| `tests/test_u2_input_lai_init.py::test_real_loader_cold_lai_threshold_and_nonvegetated_cells[8]` | 5.724 | +0.786 |
| `tests/test_u2_input_lai_init.py::test_real_loader_cold_lai_threshold_and_nonvegetated_cells[9]` | 5.724 | -0.006 |
