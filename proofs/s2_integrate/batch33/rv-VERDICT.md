# MERGE — t-camrad

Reviewed final head: `99408e7973719c47aa7b2f2251dc343a77b866e0` (rebased product `560b55fae`, original product `1e2ae174e`). This verdict admits the narrow CAM caller correction; final coupled c28 qualification follows through the manager.

The two changed product files forward the existing static non-MYNN cloud policy to CAM operand assembly through SW heating, held LW heating and land forcing. Independent AST normalization reproduces the old files exactly after removing only those policy parameters/keywords. All other product source, RRTMG paths and the entire spectral physics tree are byte unchanged. Final head's whole `src` and `tests` Git trees equal the independently measured `ae16a127e` snapshot.

## WRF caller audit

- `module_check_a_mundo.F:1802–1820` resets `icloud_bl=1` to zero outside MYNN. In `module_radiation_driver.F:1321–1431`, `cal_cldfra1` computes resolved CLDFRA; only an enabled BL-cloud policy replaces it after timestep 1 and adds QC_BL/QI_BL under the original thresholds. CAM must receive the resolved fraction under BL9. The new forwarding selects that branch, including the actual operational dispatch, and preserves the MYNN default.
- `radiation_driver.F:1976–2024` and `:2537` pass CLDFRA and QC/QI/QS to CAM. Fraction is dimensionless; moisture is a dry-air mixing ratio. `module_ra_cam.F:569–586` converts in REAL to specific water and `(QI+QS)/(1+QV)`, then reverses levels into CAM's r8 work arrays. This path is unchanged. Independent adversaries with nonzero unused QC_BL and QI_BL confirm neither leaks into non-MYNN CAM at lead 0 or 3600 s.
- `first_rk_step_part1.F:280` supplies hydrostatic P/P8W (Pa), phy_prep T (K) and PI. The unchanged operand builder follows that mapping; the production CAM-UW RK source route leaves thermodynamic tendencies for RK rather than applying them to subsequent driver inputs. No carry/time-level ordering changes are present in this fix.
- CAM derives its liquid/ice radii internally in `param_cldoptics_calc`/`reltab`/`reitab`; its caller does not receive Thompson re_cloud/re_ice/re_snow. Internal radii are micrometres and water paths g/m². `use_mp_re=0` is therefore appropriate. These calculations, latitude/monthly OZMIXM/PIN, CLWRF gases and the CAM background aerosol initialization/optics are unchanged and retain their pristine CAM01 oracle evidence.
- Existing radiation cadence, absorptivity carry/restart and land-versus-heating clock conventions are unchanged. Caller tests and same-input history reconstruction are component evidence, not proof of the coupled trajectory or history phase.

## Independent gates

[M CPU] RC01: **258 PASS, 0 FAIL, 0 SKIP** (30.90 s): all 15 original cloud/CAM caller tests and 243 shared dispatch/CLI/catalog/binding/checkpoint/NetCDF/contracts/fail-closed checks. RC06: **4 PASS** (32.07 s), adding the two active-SGS water/ice adversaries and two operational CAM spectral smokes. Total **262 PASS**.

[M CPU] RC02: independent in-process deletion of the three operational CAM policy forwards produces **one genuine numeric AssertionError, zero setup errors**. All 36 cloud elements become zero instead of one. This mutant targets the real dispatch rather than only the direct operand helper.

[M CPU trace] FULL release identities independently reproduced: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`. These are native-facade JAXPR identities, not CUDA execution claims.

[M source/data] Original cloud subroutine/source hashes match pristine WRF. Final qualification packet's **25** archived receipt hashes, decompressed hashes and raw byte counts verified directly from Git. Retained real-c28 cloud arrays: all finite, **484/484** levels within the original input-ULP envelope; three branches covered. Source/test byte identity permits reuse without another model run.

## Cost and limits

[M CPU, producer receipt independently source-bound] On real 44×42×42 initial operands, median builder wall **0.4434→0.8989 ms**; temporary bytes **1,251,024→1,871,952** (+620,928); compile **0.6190→0.6346 s**; output bytes unchanged. The initial state is clear. This cost pays for the previously omitted resolved-cloud diagnosis. The diff adds no spectral solve, persistent carry, baked asset, host transfer or synchronization; GPU launches, whole-program memory/readiness and wall cost remain [U]. No GPU performance claim follows from these CPU timings.

No merge-blocking issue found in the submitted scope. The manager must still run the final c28 24-hour comparison against original CPU-WRF under frozen D6 limits, plus the applicable GPU/E41/resource checks. Other history fluxes may be fallback RRTMG diagnostics; they do not establish CAM spectral parity. Existing XICE/SNOW and experimental CAM disclosures remain in force.

All reviewer experiments are logged on main. Disposable detached snapshot and pytest/identity scratch removed; cache NONE. Raw reviewer artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-camrad/`.
