# v0.4.0 experimental opt-in E41 supplement

The 2026-10-16 04:36Z decision keeps E41 release-blocking for default WN3/PROD/Swiss F20 programs. Opt-in composites require strict original-CPU D6, complete inventory and NF0; passing that gate with resource failures is labelled **E41 FAIL (perf/robustness), D6 PASS**. The raw E41 verdict stays FAIL. Incomplete D6 remains incomplete, and F2 justification remains separate. This applies to resolved composites, not every combination using the same scheme code.

The dated admission statements below describe their historical sources; current final-source disposition is in the linked ledger.

For the complete current provisional case inventory, read the
[GPU_PROGRESS E41 ledger](E41_LEDGER_v0.4.0.md). This supplement retains
kernel/source attribution for the exact executables named below; its older
counts are historical, not a complete final-release inventory.


Inventory cutoff: **2026-10-15T06:27:22.750143+00:00**. These rows describe exact retained GPU
executables. They do not waive the release-default hard gate or qualify
a different scheme combination. All values below are measured static
artifact properties [M]. Per-fusion runtime impact is **unknown [U]** for
every row: no matched per-kernel profile was retained for these programs.

STACK is bytes. kInput size is **optimized HLO instructions**, not SASS.
Limits remain STACK ≤256 B, kInput transpose ≤500 HLO and non-leaf callee
≤1,000 SASS. C15 has three non-leaf callee violations; other listed receipts have none.
Large caller roots with small leaf callees are not non-leaf failures.

## Configuration scope and independent CPU-WRF status

Options are MP/CU/PBL/SFCLAY/LSM/LW/SW. D6 status is copied from the
producer, without upgrading missing coverage to PASS. Full D6 PASS and
raw finiteness do not justify F2 negative events.

| Case | Options | Source | D6 | Paired frames | Program max STACK B | Flagged kernels/fusions |
|---|---|---|---|---:|---:|---:|
| mp24 | 24/0/5/5/4/4/4 | `8afb83d7dd` | PASS | 25 | 1216 | 1 |
| mp26 | 26/0/5/5/4/4/4 | `8afb83d7dd` | PASS | 25 | 304 | 2 |
| mp40 | 40/0/5/5/4/4/4 | `8afb83d7dd` | INCOMPLETE | 25 | 16 | 3 |
| c03 | 2/2/2/2/4/4/2 | `1612acc495` | PASS | 25 | 2456 | 6 |
| c04 | 3/3/3/3/4/4/1 | `e0b7697a31` | FAIL | 25 | 720 | 3 |
| c05 | 4/5/7/1/4/4/4 | `e0b7697a31` | FAIL | 25 | 744 | 3 |
| c06 | 6/6/8/1/4/4/4 | `e0b7697a31` | FAIL | 25 | 344 | 1 |
| c07 | 9/16/11/1/4/4/4 | `e0b7697a31` | INCOMPLETE | 12 | 784 | 1 |
| c08 | 10/14/12/1/4/4/4 | `e0b7697a31` | INCOMPLETE | 1 | 1472 | 9 |
| c09 | 13/94/99/1/4/4/4 | `e0b7697a31` | FAIL | 25 | 744 | 2 |
| c12 | 17/0/5/5/4/4/4 | `a939696ea6` | INCOMPLETE | 25 | 504 | 3 |
| c16 | 22/0/5/5/4/4/4 | `a939696ea6` | INCOMPLETE | 25 | 16 | 4 |
| mp18 | 18/0/5/5/4/4/4 | `8afb83d7dd` | INCOMPLETE | 25 | 504 | 3 |
| mp18_cu5 | 18/5/5/5/4/4/4 | `8afb83d7dd` | INCOMPLETE | 25 | 16 | 2 |
| c02 | 1/1/1/1/4/1/1 | `5c0e7c11ed` | PASS | 25 | 8 | 6 |
| mp28 | 28/0/5/5/4/4/4 | `8afb83d7dd` | INCOMPLETE | 25 | 248 | 4 |
| c17 | 24/93/5/5/4/4/4 | `3fd61797e8` | PASS | 25 | 32 | 1 |
| c61 | 97/0/0/91/4/4/4 | `d2782882bd` | PASS | 25 | 8 | 1 |
| c08-short | 10/14/12/1/4/4/4 | `5c0e7c11ed` | NOT_RUN | not scored | 408 | 6 |
| c15 | 21/0/5/5/4/4/4 | `afcdb5072c` | INCOMPLETE | 25 | 5920 | 1 |
| mp18-r3 | 18/0/5/5/4/4/4 | `5abde3224a` | PASS | 25 | 1336 | 7 |
| mp18_cu5-r3 | 18/5/5/5/4/4/4 | `5abde3224a` | PASS | 25 | 1336 | 8 |
| c26-qnse | 8/0/4/4/4/4/4 | `4dba72ce4c` | PASS | 25 | 7232 | 2 |

## Support-matrix disclosure rows

Every listed configuration remains **experimental: E41 STACK** and/or
**experimental: E41 kInput** and/or **experimental: E41 nonleaf** as measured. Source
Fusion sites come from rich HLO stack frames. Opaque Pallas sites show
actual custom-call stacks and explicitly verified paid-source driver bindings.
Full source distributions and additional frames are in the sealed maps.

| Case | Kernel/fusion | STACK B | kInput HLO >500 | Source sites | Nonleaf callee SASS >1000 |
|---|---|---:|---:|---|---|
| mp24 | `input_compare_minimum_reduce_select_fusion` | 1216 | — | `microphysics_wsm7.py:852` (wsm7_run); `microphysics_wsm6.py:75` (_diffac) | — |
| mp26 | `input_and_compare_multiply_select_transpose_fusion` | 304 | 945 | `microphysics_wdm7.py:1159` (wdm7_run); `microphysics_wdm7.py:939` (_wdm7_minor_loop) | — |
| mp26 | `input_maximum_minimum_select_transpose_fusion` | 192 | 874 | `microphysics_wdm7.py:1159` (wdm7_run); `microphysics_wdm6.py:137` (_qsat1_only) | — |
| mp40 | `loop_add_and_compare_divide_power_select_subtract_fusion` | 16 | 820 | `microphysics_morrison_aero.py:635` (morrison_aero_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| mp40 | `loop_and_divide_minimum_multiply_or_select_fusion` | 0 | 2227 | `microphysics_morrison_aero.py:635` (morrison_aero_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| mp40 | `input_add_and_clamp_compare_is_finite_select_transpose_fusion` | 0 | 832 | `microphysics_morrison_aero.py:635` (morrison_aero_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c03 | `input_add_compare_divide_maximum_minimum_multiply_negate_select_subtract_transpose_fusion` | 392 | 917 | `microphysics_lin.py:606` (lin_run); `_lin_update.py:170` (_conservation_and_update) | — |
| c03 | `loop_exponential_maximum_multiply_select_fusion` | 2032 | — | `ra_sw_gsfc.py:472` (_deledd); `ra_sw_gsfc.py:489` (_sagpol) | — |
| c03 | `loop_select_fusion_716` | 2456 | — | `ra_sw_gsfc.py:472` (_deledd); `ra_sw_gsfc.py:470` (_deledd) | — |
| c03 | `loop_and_multiply_select_fusion_1` | 360 | — | `ra_sw_gsfc.py:489` (_sagpol); `ra_sw_gsfc.py:483` (_sagpol) | — |
| c03 | `loop_and_multiply_select_fusion` | 360 | — | `ra_sw_gsfc.py:489` (_sagpol); `ra_sw_gsfc.py:483` (_sagpol) | — |
| c03 | `loop_add_divide_exponential_select_fusion` | 1592 | — | `ra_sw_gsfc.py:489` (_sagpol); `ra_sw_gsfc.py:483` (_sagpol) | — |
| c04 | `input_divide_multiply_reduce_fusion` | 440 | — | `microphysics_wsm3.py:104` (_nislfv_wsm3.<locals>.iterate); `microphysics_wsm6.py:356` (_precip_out) | — |
| c04 | `input_divide_multiply_reduce_fusion_1` | 720 | — | `microphysics_wsm3.py:353` (wsm3_run); `scan_adapters.py:1243` (gf_adapter) | — |
| c04 | `input_select_transpose_fusion_2` | 72 | 543 | `_gf_jax.py:243` (cup_up_aa0); `_gf_jax.py:233` (_argmax_range) | — |
| c05 | `loop_log_subtract_fusion` | 744 | — | `sfclay_revised_mm5.py:113` (_table_lookup); `sfclay_revised_mm5.py:112` (_table_lookup) | — |
| c05 | `input_transpose_fusion_342` | 0 | 1165 | `_grell_cup_jax.py:1352` (g3drv_tile); `_grell_cup_jax.py:1342` (g3drv_tile) | — |
| c05 | `input_compare_multiply_select_subtract_transpose_fusion` | 0 | 595 | `microphysics_wsm5.py:382` (wsm5_run); `microphysics_wsm6.py:107` (_qsat_fpvs) | — |
| c06 | `input_reduce_select_subtract_fusion` | 344 | — | `microphysics_wsm6.py:1088` (wsm6_run); `microphysics_wsm6.py:362` (_precip_out) | — |
| c07 | `loop_log_multiply_slice_subtract_fusion` | 784 | 1348 | `sfclay_revised_mm5.py:113` (_table_lookup); `sfclay_revised_mm5.py:112` (_table_lookup) | — |
| c08 | `input_minimum_multiply_reduce_select_fusion` | 1472 | 786 | `microphysics_morrison.py:666` (morrison_tendency); `scan_adapters.py:154` (_mp_in) | — |
| c08 | `input_clamp_divide_subtract_transpose_fusion` | 288 | 588 | `sfclay_revised_mm5.py:114` (_table_lookup); `sfclay_revised_mm5.py:88` (_psih_unstable_full) | — |
| c08 | `loop_and_divide_minimum_multiply_select_slice_fusion` | 0 | 508 | `native_sas_adapter.py:102` (native_sas_cadence_step); `surface_layer.py:700` (_surface_layer_impl) | — |
| c08 | `loop_add_and_compare_divide_multiply_or_power_select_fusion` | 0 | 883 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c08 | `loop_add_and_clamp_multiply_power_select_subtract_fusion` | 0 | 1386 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c08 | `loop_minimum_multiply_select_fusion` | 0 | 766 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c08 | `input_minimum_multiply_select_transpose_fusion` | 0 | 526 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:72` (_flatau) | — |
| c08 | `loop_add_minimum_multiply_select_fusion` | 0 | 533 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c08 | `input_convert_select_transpose_fusion` | 0 | 510 | `microphysics_morrison.py:666` (morrison_tendency); `scan_adapters.py:2073` (_fp64_island.<locals>.island) | — |
| c09 | `loop_log_subtract_fusion` | 744 | — | `sfclay_revised_mm5.py:113` (_table_lookup); `sfclay_revised_mm5.py:112` (_table_lookup) | — |
| c09 | `loop_select_fusion_95` | 0 | 682 | `surface_layer.py:209` (_table_lookup); `surface_layer.py:208` (_table_lookup) | — |
| c12 | `input_and_multiply_power_reduce_select_fusion` | 504 | — | `mathfun.py:60` (gamxinfdp); `mathfun.py:84` (gamxinfdp) | — |
| c12 | `input_add_convert_select_transpose_fusion` | 0 | 665 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_c.py:609` (gs_part_c) | — |
| c12 | `input_add_and_transpose_fusion` | 0 | 728 | `gs_c.py:693` (gs_part_c.<locals>.vap_i); `gs_c.py:697` (gs_part_c.<locals>.vap_d) | — |
| c16 | `input_add_and_compare_divide_or_select_transpose_fusion` | 0 | 923 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_a.py:184` (gs_part_a) | — |
| c16 | `input_add_and_select_transpose_fusion` | 0 | 1109 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_a.py:488` (_post_setvtz.<locals>.<lambda>) | — |
| c16 | `input_add_and_compare_divide_maximum_minimum_multiply_select_slice_transpose_fusion` | 0 | 627 | `gs_c.py:693` (gs_part_c.<locals>.vap_i); `gs_c.py:919` (gs_part_c) | — |
| c16 | `input_and_or_reduce_select_fusion_1` | 0 | 514 | `gs_c.py:919` (gs_part_c); `gs_c.py:941` (gs_part_c) | — |
| mp18 | `input_and_multiply_power_reduce_select_fusion` | 504 | — | `mathfun.py:60` (gamxinfdp); `mathfun.py:84` (gamxinfdp) | — |
| mp18 | `input_add_convert_select_transpose_fusion` | 0 | 665 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_c.py:609` (gs_part_c) | — |
| mp18 | `input_add_and_transpose_fusion` | 0 | 728 | `gs_c.py:693` (gs_part_c.<locals>.vap_i); `gs_c.py:697` (gs_part_c.<locals>.vap_d) | — |
| mp18_cu5 | `input_add_convert_select_transpose_fusion` | 0 | 665 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_c.py:609` (gs_part_c) | — |
| mp18_cu5 | `input_add_and_transpose_fusion` | 0 | 728 | `gs_c.py:693` (gs_part_c.<locals>.vap_i); `gs_c.py:697` (gs_part_c.<locals>.vap_d) | — |
| c02 | `input_add_clamp_convert_divide_gather_select_transpose_fusion` | 0 | 570 | `ra_lw_rrtm_jax.py:220` (_fint); `ra_lw_rrtm_jax.py:358` (_prepare_atmosphere_jax) | — |
| c02 | `loop_select_fusion_19` | 0 | 549 | `ra_lw_rrtm_jax.py:233` (_row); `ra_lw_rrtm_jax.py:260` (_binary_lower) | — |
| c02 | `loop_select_fusion_20` | 0 | 661 | `ra_lw_rrtm_jax.py:233` (_row); `ra_lw_rrtm_jax.py:249` (_interp4) | — |
| c02 | `loop_select_fusion_21` | 0 | 659 | `ra_lw_rrtm_jax.py:233` (_row); `ra_lw_rrtm_jax.py:249` (_interp4) | — |
| c02 | `loop_broadcast_transpose_fusion` | 0 | 586 | `ra_lw_rrtm_jax.py:233` (_row); `ra_lw_rrtm_jax.py:249` (_interp4) | — |
| c02 | `input_subtract_transpose_fusion_1` | 0 | 632 | `sfclay_revised_mm5.py:114` (_table_lookup); `sfclay_revised_mm5.py:88` (_psih_unstable_full) | — |
| mp28 | `input_add_maximum_select_transpose_fusion` | 0 | 833 | `physics_couplers.py:2131` (thompson_aero_adapter); `thompson_aero_column.py:461` (_eff_aero) | — |
| mp28 | `input_add_divide_minimum_multiply_reduce_select_sqrt_subtract_fusion` | 0 | 827 | `physics_couplers.py:2131` (thompson_aero_adapter); `physics_couplers.py:2655` (mynn_adapter_with_source_leaves) | — |
| mp28 | `input_add_divide_maximum_minimum_power_select_subtract_transpose_fusion` | 248 | 1175 | `physics_couplers.py:2131` (thompson_aero_adapter); `thompson_column.py:735` (_snow_moment) | — |
| mp28 | `input_add_and_compare_convert_divide_multiply_power_transpose_fusion` | 40 | 739 | `physics_couplers.py:2131` (thompson_aero_adapter); `thompson_column.py:450` (density_from_pressure_temperature) | — |
| c17 | `input_compare_reduce_select_fusion` | 32 | 844 | `microphysics_wsm7.py:852` (wsm7_run); `microphysics_wsm7.py:656` (_wsm7_minor_loop) | — |
| c61 | `loop_add_compare_divide_maximum_minimum_multiply_power_select_sqrt_subtract_fusion` | 0 | 527 | `microphysics_goddard.py:469` (_saticel_cell); `microphysics_goddard.py:643` (_saticel_cell) | — |
| c08-short | `loop_maximum_minimum_multiply_select_subtract_fusion` | 408 | 1320 | `sfclay_revised_mm5.py:113` (_table_lookup); `sfclay_revised_mm5.py:112` (_table_lookup) | — |
| c08-short | `loop_add_divide_multiply_select_subtract_fusion` | 0 | 1334 | `sfclay_revised_mm5.py:113` (_table_lookup); `sfclay_revised_mm5.py:112` (_table_lookup) | — |
| c08-short | `loop_and_divide_multiply_select_subtract_fusion` | 0 | 843 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c08-short | `loop_add_and_clamp_compare_minimum_multiply_or_power_select_subtract_fusion` | 0 | 1630 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:154` (gamma_fn) | — |
| c08-short | `loop_add_compare_multiply_select_fusion` | 0 | 974 | `microphysics_morrison.py:666` (morrison_tendency); `microphysics_morrison.py:72` (_flatau) | — |
| c08-short | `loop_minimum_multiply_select_fusion` | 0 | 734 | `microphysics_morrison.py:666` (morrison_tendency); `scan_adapters.py:156` (_mp_in) | — |
| c15 | `input_add_and_compare_convert_divide_maximum_minimum_multiply_power_reduce_select_slice_subtract_fusion` | 5920 | — | `gs_a.py:571` (_post_setvtz); `gs_a.py:488` (_post_setvtz.<locals>.<lambda>) | 10240 / 8899 / 13137 |

| mp18-r3 | `input_add_and_divide_maximum_minimum_multiply_select_subtract_transpose_fusion` | 1336 | 1172 | `gs_c.py:402` (gs_part_c); `gs_c.py:400` (gs_part_c) | — |
| mp18-r3 | `input_add_compare_convert_maximum_minimum_select_subtract_transpose_fusion` | 0 | 1128 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_b.py:958` (_deposition) | — |
| mp18-r3 | `input_compare_divide_minimum_multiply_select_transpose_fusion` | 0 | 662 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_a.py:488` (_post_setvtz.<locals>.<lambda>) | — |
| mp18-r3 | `input_add_and_compare_divide_minimum_select_transpose_fusion` | 0 | 835 | `gs_a.py:194` (gs_part_a.<locals>.<lambda>); `gs_c.py:497` (gs_part_c.<locals>.sc) | — |
| mp18-r3 | `input_add_select_transpose_fusion_1` | 0 | 562 | `gs_c.py:745` (gs_part_c.<locals>.srr); `gs_a.py:488` (_post_setvtz.<locals>.<lambda>) | — |
| mp18-r3 | `input_add_or_select_subtract_transpose_fusion` | 0 | 1041 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_a.py:184` (gs_part_a) | — |
| mp18-r3 | `input_add_select_transpose_fusion` | 0 | 724 | `mathfun.py:84` (gamxinfdp); `mathfun.py:60` (gamxinfdp) | — |
| mp18_cu5-r3 | `input_add_and_divide_maximum_minimum_multiply_select_subtract_transpose_fusion` | 1336 | 1172 | `gs_c.py:402` (gs_part_c); `gs_c.py:400` (gs_part_c) | — |
| mp18_cu5-r3 | `input_and_convert_minimum_multiply_select_transpose_fusion` | 48 | 1064 | `mathfun.py:60` (gamxinfdp); `mathfun.py:84` (gamxinfdp) | — |
| mp18_cu5-r3 | `input_add_compare_convert_maximum_minimum_select_subtract_transpose_fusion` | 0 | 1134 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_b.py:958` (_deposition) | — |
| mp18_cu5-r3 | `input_compare_divide_minimum_multiply_select_transpose_fusion` | 0 | 660 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_a.py:488` (_post_setvtz.<locals>.<lambda>) | — |
| mp18_cu5-r3 | `input_add_and_compare_divide_minimum_select_transpose_fusion` | 0 | 835 | `gs_a.py:194` (gs_part_a.<locals>.<lambda>); `gs_c.py:497` (gs_part_c.<locals>.sc) | — |
| mp18_cu5-r3 | `input_add_select_transpose_fusion_1` | 0 | 562 | `gs_c.py:745` (gs_part_c.<locals>.srr); `gs_a.py:488` (_post_setvtz.<locals>.<lambda>) | — |
| mp18_cu5-r3 | `input_add_or_select_subtract_transpose_fusion` | 0 | 1041 | `gs_b.py:120` (gs_part_b.<locals>.<lambda>); `gs_a.py:184` (gs_part_a) | — |
| mp18_cu5-r3 | `input_add_select_transpose_fusion` | 0 | 724 | `mathfun.py:84` (gamxinfdp); `mathfun.py:60` (gamxinfdp) | — |
| c26-qnse | `qnse_pbl_columns` | 7232 | — | `phys_qnse_columns.py:98` (qnse_columns); `pbl_qnse.py:222` (qnsepbl_column) | — |
| c26-qnse | `qnse_mf_columns` | 2952 | — | `phys_qnse_columns.py:98` (qnse_columns); `qnse_edmf.py:271` (mfshconvpbl_column) | — |

## Remedy disposition and resource consequences

No production code change is included. The production `src/gpuwrf` tree
is byte-identical to the lane base; default arithmetic, launches, traffic,
temporary allocations, peak VRAM and executable construction are unchanged.
No speedup, roofline closure or isolated memory-impact claim is made.

The cheapest public-output firewall was screened on the real Swiss inputs
through `MP_SCAN_ADAPTERS`, including its production fp64 island, State
replacements, precipitation accumulation and theta conversion: MP24
**164/164** and MP26 **158/158** State leaf results byte-identical over two
successive MP calls, all finite. These are CPU regression/neutrality checks;
dynamics and other physics were not stepped in this local screen. They
are not WRF fidelity or native-resource proof. The public firewall is not
shipped: its effect on the internal reduction/transpose mega-fusions has
not been established on a whole-root native executable. Internal cuts
would add materialization/traffic and launches and require fresh native
measurements; fixed-loop rewrites also require the same byte gate.

C03 E41 disclosure/remedies belong to `t-e41opt` by manager transfer.
SF1/PBL numerical files retain their owners; they were not edited. Their retained failures are
listed for complete disclosure, without claiming a fix or transferring
runtime between similarly named fusions. MP40 bounded D6 passed, but its
full comparison is INCOMPLETE: 12 CPU fields are missing. `t-writer` owns
that independent coverage issue.

EO03, the initial whole-root CPU harness, reused a donated buffer and
failed before candidate execution. That setup failure is preserved and
logged; it is neither a candidate numerical failure nor a passing gate.

## Evidence

Active inventory/maps are in `proofs/t_e41opt_updates/c26_qnse/`; earlier sealed packets remain unchanged.
`INVENTORY.json` binds every producer gate and input manifest by SHA256.
Producer `e41/pins` retain actual native executables and complete scans;
the attribution maps bind those PINs, decompressed blobs and rich HLO.
`SOURCE_MAPS.json` preserves the current compact attribution evidence;
`proofs/t_e41opt/CPU_NEUTRALITY.json` retains the original MP24/26 byte checks.
Full maps reside at the exact paths recorded in the inventory and source maps,
including `<DATA_ROOT>/wrf_gpu2/scratch/`. Producer artifacts were read-only.
Separate firewall candidates and their pre-screen scratch have their own
receipts; no candidate is admitted or included in this disclosure change.

## Final C03 successor

The active C03 row is the final `1612acc49` / `CR29_c03` GPU arm: original CPU-WRF
D6 PASS over all 25 frames, complete coverage, zero bounded failures and raw
nonfinite values. Its six STACK flags (max 2,456 B) and Lin kInput 917 HLO
remain FAIL. The new Lin fusion uses 392 B; its source stack includes
`microphysics_lin.lin_run`, `_lin_update._conservation_and_update`,
`microphysics_lin._process_block` and `_lin_graupel._graupel_block`.

The earlier v-prep C03 five-flag / D6 FAIL arm is historical evidence, preserved
unchanged in the original packet at commit `8c4ede6fc`, and under `historical_c03`
in the successor inventory. It is not relabelled as passing. At that update, the inventory
had 12 configurations and 38 flags; the MP18 update increased that to 14 receipts and 43 flags. No Lin/GSFC firewall is shipped.

**Source-family correction:** the five older stack fusions are dominated by
GSFC shortwave `ra_sw_gsfc._deledd` / `_sagpol`, not Eta surface code. The
exact rich-HLO body census attributes 1,979/2,095; 1,842/1,933; 964/1,103;
964/1,103; and 1,652/1,723 instructions to `ra_sw_gsfc.py`. These counts
are static source attribution, not runtime weights. `proofs/t_e41opt_updates/c03_final/SOURCE_FAMILIES.json`
binds the complete body census to the retained blob and rich-HLO SHA.

The separate MP24/26 output-firewall branch `lane/t-e41opt-firewall` (`c4c9ddb65`)
has 322/322 CPU caller results BYTE and exact Swiss/PROD default identities,
but remains NOT ADMITTED pending captured-target native STACK clearance and
the next actual MP24/MP26 GPU arms. It is not part of this disclosure update.

## MP18 successor

Both completed MP18 arms retain 25-frame bounded D6 PASS and zero raw
nonfinite values, but full coverage is INCOMPLETE: `REFD_MAX` and `REFL_10CM`
are missing. No diagnostic is zero-filled and no F2 event justification is inferred.
MP18 has STACK 504 B plus kInput 665/728 HLO. MP18+CU5 has the two 665/728 HLO
flags, with no STACK-over-256 failure. Resource properties and source maps were
read from each exact executable; the 504 B frame is not transferred by name
to the CU5 program. Both GPU processes completed before the 15:07Z STOP.
These are distinct selector/source receipts; legacy MP17 (C12) normalizes to MP18.

## Final C02 and MP28 successor

Final C02 is the `5c0e7c11e` / `TCN23_c02` corrected GPU arm, with original
CPU-WRF D6 PASS over all 25 frames and complete coverage. E41 still FAILs:
six kInput fusions have 570/549/661/659/586/632 optimized HLO instructions.
Its maximum STACK is 8 B (PASS), and nonleaf-over-1k count is zero. The
earlier nonfinite C02 arm and its 752 B STACK frame remain historical;
their properties are not transferred to the corrected executable.

MP28 (`8afb83d7d`) retains bounded D6 PASS for 25 frames and zero raw
nonfinite values, but full coverage is INCOMPLETE: `QNBCA`, `QNIFA2D`,
`QNWFA2D`, and `TAOD5502D` are missing. Its four kInput fusions have
833/827/1175/739 HLO instructions. Maximum STACK 248 B and nonleaf zero
pass. Both arms completed after GPU RESUME; neither is an S1 benchmark.

The C02/MP28 update brought the inventory to 16 source/selector receipts and 53 flags.
No SF1/PBL numerical code was edited. No E41 waiver or runtime-cost
claim follows from source attribution, and no firewall is shipped.

V-prep C13 and C19 reuse these exact MP18+CU5 and MP28 native receipts,
respectively. Their own matched original CPU-WRF rescoring is also
INCOMPLETE over 25 frames with no bounded failures or nonfinite values.
`proofs/t_e41opt_updates/c02_mp28/ALIAS_RECEIPTS.json` binds both reuse decisions and D6 reports. They
retain the corresponding E41 FAILs and add no new native realization
or duplicated flags to the count above.

## Separate MP24 firewall pre-screen disposition

The separate `lane/t-e41opt-firewall` candidate is NOT ADMITTED. Its
captured-RTX-target CPU native pre-screen reproduced baseline STACK
1,216 B, then measured candidate STACK 9,616 B and 20 nonleaf-over-1k
findings. Cold compile elapsed was 237.250 s versus 740.405 s; native
size was 23,837,107 B versus 32,000,643 B. Both static XLA allocation
reports gave 271,095,137 B; this is not peak VRAM. These are measured
CPU pre-screen properties, not GPU throughput or actual MP options
replay: the paired control used captured CAM options with autotune zero.
Semantic options were byte-equal after removing two private path fields.

The negative packet is retained only on that candidate branch at
`39abddfd4`, in `proofs/t_e41opt_firewall/native_prescreen/`. No GPU arm
was queued and no MP26 native arm was spent on this failed combined
candidate. Its caches and detached snapshot were cleaned after E41.
The separate C03 Lin/GSFC candidate remains CPU-neutrality-only and
NOT ADMITTED; native/default/actual-GPU resource gates remain pending.

## C17 MP24/CU93 successor

C17 is the actual `3fd61797e` MP24/CU93 executable: full original
CPU-WRF D6 PASS over 25 frames, zero bounded failures and nonfinite
values. E41 FAIL is solely kInput 844 HLO in
`input_compare_reduce_select_fusion`; the rich source stack resolves
to WSM7 `wsm7_run` / `_wsm7_minor_loop`. Program maximum STACK 32 B
and nonleaf zero pass. This distinct source/selector executable does
not inherit the 1,216 B frame of the earlier MP24/CU0 program. No
firewall candidate or native pre-screen property is substituted for
this actual GPU result. Runtime impact is unknown.

The C17 update brought the inventory to 17 source/selector receipts and 54 flagged fusions.
No production code or producer reference was changed. The one new
CPU attribution arm ran under a 6 GiB memory cap with zero swap,
on lane cores with JAX CPU only; it created no compile cache or snapshot.

## Post-reboot disclosure reconciliation

C02 is already disclosed: six kInput 570/549/661/659/586/632 failures,
maximum STACK 8 B and nonleaf zero PASS, full original CPU-WRF D6 PASS
over 25 frames.
No C02 forecast or numerical gate was repeated.

C61 (`d2782882bd`, MP97/PBL0/SF91) has full D6 PASS over 25 frames with
zero bounded failures/nonfinite values. E41 FAIL is solely kInput 527 HLO,
in the Goddard `_saticel_cell` source family. Maximum STACK 8 B and
nonleaf zero pass. Its artifact remains distinct from later source trees.

C08-short is `TCN24_c08` / `5c0e7c11e`, a failed intended 2h arm. It
terminated with NonFiniteStateError at step 200 (3,600 s) and retained
only lead 0 output. No D6 scoring or complete forecast proof exists. Its
actual JIT+AOT scans show STACK 408 B and six kInput sizes
1320/1334/843/1630/974/734 HLO instructions.
The first two source stacks resolve to SF1 table lookup; the other four
to Morrison. The earlier C08 arm remains unchanged. Neither native
properties nor the nonfinite cause are transferred between sources.

C15 (`afcdb5072c`, MP21) completed 25 frames with bounded D6 PASS and
zero nonfinite values, but full coverage is INCOMPLETE: REFD_MAX and
REFL_10CM are missing. E41 FAIL includes STACK 5,920 B and three nonleaf
callees of 10,240/8,899/13,137 SASS instructions, with 14/8/16 nested calls.
The parent fusion resolves to NSSL source sites. The native
STACK label is truncated; its unique full cubin kernel and rich HLO
fusion name are joined explicitly in the sealed evidence. The source
sites describe the parent fusion; individual native callees are not
assigned source or runtime weights by this mapping.

The post-reboot update covered 20 source/selector receipts, 62 flagged fusions and 3
nonleaf callee findings. Runtime impact remains unknown. No remedy or
firewall is admitted. All work used retained pre-reboot artifacts;
there was no GPU work, forecast rerun, compile cache or snapshot.

## NSSL round3: radar carry and STACK comparison

Both actual `5abde3224` arms have full original CPU-WRF D6 and coverage
PASS: 25 frames, 372 CPU fields, 62 bounded fields and zero raw nonfinite
values. REFL_10CM is inventoried and finite but REPORT_ONLY under the
frozen manifest. F2 negative-event justification remains OPEN. These
facts do not qualify a different source or waive E41.

E41 FAIL remains in both arms: maximum STACK 1,336 B, seven kInput
failures for MP18 and eight for MP18+CU5 (maximum 1,172 HLO), nonleaf
zero. The STACK fusion is itself the 1,172-HLO kInput failure and is
counted once per executable. Older round2 records/maps remain unchanged.

**Measured increase:** MP18 program maximum STACK rose from 504 B to
1,336 B: +832 B, or 2.65 times. MP18+CU5 previously had maximum 16 B
and no STACK failure; it now also has 1,336 B. These are different
fusions: the earlier maximum was `input_and_multiply_power_reduce_select_fusion`
(chiefly `mathfun.gamxinfdp`), while the new maximum is
`input_add_and_divide_maximum_minimum_multiply_select_subtract_transpose_fusion`.

**Radar attribution:** the new 1,172-instruction fusion has 580
instructions attributed to `gs_c.py`, 132 to `gs_a.py` and 126 to
`gs_b.py`, in both arms. None of its 880/879 attributed source stacks
contains `radardd02`. Direct `radardd02` attribution identifies nine
MP18 and eleven MP18+CU5 fusions; all match actual native kernels with
STACK 0 B, and their largest HLO body is 400 instructions.

These are static fusion counts, not launches per timestep.

Thus the measured large frame belongs to fused NSSL process work,
rather than a directly radar-attributed kernel. An indirect fusion
change caused by added radar/carry outputs is possible [I], but is
not isolated: the coupling graphs differ and no radar-only native A/B
was measured. `gs_a.py`, `gs_b.py`, `gs_c.py`, `mathfun.py` and
`diagnostics.py` are byte-identical between the old and new sources.
Static source counts are not runtime weights or a speed/memory ratio.
`proofs/t_e41opt_updates/nssl_round3/STACK_COMPARISON.json` seals the complete census and byte proof.

The NSSL round3 update covered 22 source/selector receipts, 77 flagged fusions and
three nonleaf callee findings (the latter remain the earlier C15 arm).
No GPU, forecast, numerical validation or native compile was repeated
for this disclosure. No remedy or firewall is admitted.

## C26 QNSE actual native resources

The reviewed `t-qnsecompile` packet is `17943b8b2`; the actual paid GPU
source is `4dba72ce4`, with the entire model tree byte-identical between
them. C26 has full original CPU-WRF D6 PASS over 25 frames, 62 bounded
fields and 360 fields in the complete CPU inventory: zero bounded
misses, nonfinite values or missing fields. No numerical gate was repeated.

E41 remains FAIL on the actual paid JIT+AOT executable (same native ELF):
`qnse_pbl_columns` STACK **7,232 B**, `qnse_mf_columns` STACK **2,952 B**.
QNSE surface STACK is 0 B. Nonleaf and kInput-over-500 counts are zero.
These are opaque Pallas native kernels, so their STACK values are not
optimized HLO fusion instruction counts. The actual rich-HLO custom calls
map both symbols to `phys_qnse_columns.qnse_columns`; the paid-source AST
binds PBL to `pbl_qnse.qnsepbl_column` and MF to
`qnse_edmf.mfshconvpbl_column`. The table distinguishes this source
binding from a kernel-body HLO stack frame. Standalone pre-screen
STACK 9,608/2,584 B is not substituted for the actual values above.

The compiler-stall repair is independently qualified; its actual
resource failures remain experimental opt-in disclosures. Cold/shared
24h wall 519.776 s and sampled peak VRAM 7,054 MiB are qualification
observations, not S1 or isolated kernel overhead. Runtime spill cost and
occupancy impact remain unknown without a matched native profile.
F2 negative-event and guard-repair justification remains OPEN.

Current scope: 23 source/selector receipts and 79 flagged native kernels
or HLO fusions, with three earlier C15 nonleaf callee findings. Default
E41 limits remain unchanged. This update changes no production code and
runs no GPU, forecast, native compile or repeated numerical validation.
