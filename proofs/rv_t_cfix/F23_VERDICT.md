# F23 — MERGE implemented CPU caller hunks; QNSE sea ice remains OPEN

Reviewed `2daaa8e6d3`, implementation `dad9dd5fc5`, base `50517329d8`.
Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/PM3/`; `PM09c-noah.xml`, `PM09d-ruc.xml`, `PM09d-ssib.xml`, `PM11-F23-solver-reuse.json`.

Scope approved: Noah-MP opt_sfc=2 seeds from its own prior CH/CM; RUC/SSiB receive the already-produced topographic radiation operands. **This is not closure of the entire F23 finding:** QNSE still receives SICE=0 and no XICE was implemented. Ice-free c26 does not test sea ice. The producer's proposed KNOWN_ISSUES wording is not an implemented input refusal; no sea-ice qualification/admission claim is approved. Manager should retain the open finding/disclosure.

- Pristine `module_sf_noahmpdrv.F:817–818` supplies CHXY/CMXY; SFCDIF2 iterates from that state. [M CPU] Four focused caller tests pass. An independent two-call test executes the actual Noah-MP adapter and solver and verifies the returned own CH/CM seed the next call; coefficients remain finite.
- Pristine `module_surface_driver.F:1918–1936,7137–7139` adjusts SWDOWN and scales GSW before the LSMs. The port's `swnorm` names the topographic downward flux (different naming from WRF's saved SWNORM). RUC needs `swnorm − swup_topographic`; SSiB needs `swnorm`. [M CPU] Independent two-step operational RUC and SSiB tests execute real LSM hooks/solvers and carry with slope_rad=1, held constructed radiation: 640−110=530 W/m² for RUC, 640 W/m² for SSiB on both calls. They verify the next land carry is consumed. RK/microphysics are isolated, so these are caller lifetime checks, not forecast/oracle truth.
- [M source] Noah-MP solver tree, RUC/SSiB solvers and hooks, and RRTMG-SW numerical source are BYTE-identical to base. Original solver qualifications are retained; these changes only select caller operands, whose authoritative pristine statements were checked. No new numerical solver oracle is claimed or required for unchanged equations.
- Independent restoration of sfclay seeds at opt_sfc=2 produces one assertion failure, zero setup errors/skips.
- Both default full native-facade hashes independently EXACT: Swiss f672a573 / PROD 6fc715a2. Opt_sfc=1 retains the same seed expression. With slope_rad=0, upstream topographic fields equal flat fields and the selected arrays/arithmetic are unchanged; this does not establish new GPU slope_rad=1 qualification.

[I source] No new State/carry leaves, column solver, radiation solve or host transfer. Noah seed selection is a static branch; RUC keeps its existing one subtraction, SSiB forwards one existing array. [U] Actual changed-configuration GPU launch/peak/readiness/time remain unmeasured. No negligible-cost claim. Existing c26/c33/c40 default options are regression scope only; they do not close QNSE sea ice or qualify slope_rad=1 forecasts.

Seven healthy CPU checks total (four primary + three independent lifetime checks). Initial missing fixture/import-order/precipitation-stub errors were excluded; only invalid setups were rerun. No producer edits/GPU run; snapshot/builds/caches removed. Integrate the scoped hunks separately from blocked F17 and keep F23/SICE open.
