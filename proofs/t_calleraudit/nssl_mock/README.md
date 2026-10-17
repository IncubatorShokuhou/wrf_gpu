# Pre-existing NSSL red tests on main 403240f257 — test-double repairs (zero known-red)

Lane t-calleraudit · base main 403240f257 · review `403240f257..<lane head>` · test-only (no `src` change) · CPU only.
Manager 16:06:45Z (rv-r-pkg040 scope_fix VERDICT #2: "4 NSSL failures, scan_adapters.py:2489 needs rainncv and sr").

## 1. Defects [M CPU, main 403240f257, one file per 6G RAM slot: `BASE_main403240f257.txt`]
| tests | error | cause |
|---|---|---|
| `tests/v040/test_nssl_one_moment_wiring.py::test_actual_post_rk_variant_and_cold_start` [mp19/21 × step1/2] (the 4 named) | `KeyError: 'rainncv'` at `scan_adapters.py:2489` | `nssl2mom_adapter` double returns the pre-F1 dict |
| `tests/v040/test_nssl_nohail22.py::test_operational_post_rk_selects_no_hail_and_first_call` [step1/2] (not in the 14-file subset) | same | same |
| `tests/v034/nssl2mom/test_radius_handoff.py::test_actual_adapter_holds_pristine_radii_and_disable_stays_absent` (red when the file runs alone) | `TracerArrayConversionError` (sfclay_revised_mm5.py:97 import-time table) | adapter.py:63 lazily imports `scan_adapters` (beb000a9b2); first call under `jax.jit` runs that import, and sfclay_revised_mm5's import-time jnp tables are traced |

Attribution of the first two rows: my own F1 93dfdac853 (consumer reads `rainncv`) and F2-fix+F24 76860e2d14 (`sr`) changed
`nssl_public_precipitation`; F1 updated `nssl_cumulus/test_caller.py` but missed these two doubles. The real producer returns
`rainncv, snowncv, grplncv, hailncv, sr` (nssl2mom/driver.py:100) + `rain_liquid` (:166).
Production is not affected by row 3: operational_mode.py:129 imports `scan_adapters` at module level before any trace.

## 2. Fix (test-only)
- c8ebdeb64b: both post-RK adapter doubles return the producer's six keys (same set the main `test_caller` /
  `test_radar_handoff` doubles use). Column-level `nssl2mom_column` stubs are untouched (the adapter's accumulators read only
  rain_liquid/snowncv/grplncv/hailncv; they pass).
- f4fe2fad10: `test_radius_handoff.py` imports `gpuwrf.coupling.scan_adapters` at module level (production order).
- No assert changed, no skip/xfail added.

## 3. Evidence [M CPU]
- main 403240f257: 529 PASS / 7 FAIL over 12 NSSL files (`BASE_main403240f257.txt`, signatures `BASE_error_signatures.txt`).
- head f4fe2fad10: **536 PASS / 0 FAIL** over the same 12 files (`AFTER_f4fe2fad10.txt`; runner `run_files.sh`).
- Deletion sensitivity: reverting c8ebdeb64b = main → 6 genuine `KeyError: 'rainncv'`; reverting f4fe2fad10 = main → 1 genuine
  `TracerArrayConversionError`; mutant M2 (keep rainncv, drop only `sr`) → 6 genuine `KeyError: 'sr'`, 0 setup errors
  (`MUTANT_M2_no_sr.txt`). Each test also asserts the double was called exactly once (E114).
- Combined 12-file run in ONE 6G slot is OOM-killed at ~67 % (rc 137) on main as well → run per file.

## 4. Performance / identity
No `src` change → default program unchanged by construction (no identity re-trace needed); no runtime cost.
Latent hazard (disclosed, not changed in release): a lazy import inside a function first called under jit executes import-time
jnp tables under the trace (E206 sibling). Only isolated tests reach it today.
