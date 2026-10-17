# c40 release blocker (mp8 / YSU / sf1 / SSiB sf8 / ra_lw4 / ra_sw1) — lane t-ysuqc, 2026-10-16

**Symptom** (v-prep FINAL on 04cb09f816): D6 FAIL, T2 RMSE 4.9 K (lead 8) → 15.4 K (10) → 21 K (13).
Passed on f68ca7296 (T2 ≤ 0.09 K all leads).

**Localization [M]** (`C40_LOCALIZE_cpu_vs_gpu_f68_04cb.json`, all-land domain): nights match CPU-WRF
(04cb even closer than f68); from sunrise GPU SWDOWN ≈ 0 (bias −673 W/m² at 12 UTC), then GRDFLX/HFX/LH/
TSK/T2 collapse (TSK −20.8 K at 13 UTC). GLW/SWNORM unaffected except via TSK.

**Root cause [M source + producer probe]:** `087e926beb` (F23, caller audit) changed the SSiB caller in
`runtime/operational_mode.py` to `ssib_rad = (rdiag.swnorm, …)`, assuming swnorm == swdown at slope_rad=0.
True for the RRTMG coupler (`swnorm = surface_down_topographic`), false for SSiB's own radiation
(`coupling/ssib_surface_hook.ssib_radiation`: Dudhia SW + RRTMG LW) which sets `swnorm = 0` — matching
WRF's SWNORM history at slope_rad=0 (CPU-WRF c40 SWNORM ≡ 0). Real producer at 12 UTC, 42°N:
swdown 776.6 W/m², swnorm 0.

**Fix:** WRF surface_driver.F:1918-1936 — TOPO_RAD_ADJ_DRVR (topographic SWDOWN) runs only for
slope_rad=1. SSiB now receives `rdiag.swdown`; slope_rad=1 with SSiB raises (no topographic pair on the
SSiB radiation path). This restores the f68 SSiB input for slope_rad=0.

**Gates:** `tests/test_ssib_swdown_caller.py` (real `ssib_radiation` at a daytime clock → SSiB surface step
receives its swdown; slope_rad=1 refused); F23's SSiB seam test now expects the unadjusted SWDOWN.
Mutant "revert to swnorm": both tests fail by genuine assertion, 0 setup errors. Default programs
Swiss 7379c7d7 / PROD 6487f5b0 = current main. Decisive: v-prep c40 GPU rerun + D6.
