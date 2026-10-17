Main microphysics caller audit — `ce35f684c0d8be4a82bad3963f3412d24f4f9924`

Source audit of all 29 active dispatcher codes, passive 0, and eight requested unbound codes. Nine active main callers lack both dry-theta preparation and moist-theta finish; twenty have the handoff. This is source evidence, not numerical qualification. Only Lin is changed in this lane.

| MP | State caller (main file:line) | Input → output helper | Verdict |
|---|---|---|---|
| 0 | `disabled/passive` — `src/gpuwrf/contracts/physics_registry.py:93` | N/A | PASSIVE |
| 1 | `kessler_adapter` — `src/gpuwrf/coupling/scan_adapters.py:209` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 2 | `lin_adapter` — `src/gpuwrf/coupling/scan_adapters.py:228` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 3 | `wsm3_adapter` — `src/gpuwrf/coupling/scan_adapters.py:363` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 4 | `wsm5_adapter` — `src/gpuwrf/coupling/scan_adapters.py:381` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 5 | `ferrier_adapter` — `src/gpuwrf/physics/ferrier/adapter.py:8` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 6 | `wsm6_adapter` — `src/gpuwrf/coupling/scan_adapters.py:284` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 7 | `goddard4ice_adapter` — `src/gpuwrf/coupling/goddard4ice_adapter.py:8` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 8 | `thompson_adapter` — `src/gpuwrf/coupling/physics_couplers.py:1866` | _dry_temperature_from_state -> T/pi -> _theta_m_from_microphysics | HANDOFF_PRESENT |
| 9 | `milbrandt_adapter` — `src/gpuwrf/physics/milbrandt2mom/adapter.py:13` | manual REAL old_theta/(1+Rv/Rd*qv) + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 10 | `morrison_adapter` — `src/gpuwrf/coupling/scan_adapters.py:399` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 11 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 13 | `sbu_ylin_adapter` — `src/gpuwrf/coupling/scan_adapters.py:255` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 14 | `wdm5_adapter` — `src/gpuwrf/coupling/scan_adapters.py:449` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |
| 15 | `ferrier_adapter` — `src/gpuwrf/physics/ferrier/adapter.py:8` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 16 | `wdm6_adapter` — `src/gpuwrf/coupling/scan_adapters.py:484` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 17 | `nssl2mom_adapter` — `src/gpuwrf/physics/nssl2mom/adapter.py:39` | _dry_theta_view + _temperature_from_theta(dry) + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 18 | `nssl2mom_adapter` — `src/gpuwrf/physics/nssl2mom/adapter.py:39` | _dry_theta_view + _temperature_from_theta(dry) + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 19 | `nssl2mom_adapter` — `src/gpuwrf/physics/nssl2mom/adapter.py:39` | _dry_theta_view + _temperature_from_theta(dry) + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 20 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 21 | `nssl2mom_adapter` — `src/gpuwrf/physics/nssl2mom/adapter.py:39` | _dry_theta_view + _temperature_from_theta(dry) + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 22 | `nssl2mom_adapter` — `src/gpuwrf/physics/nssl2mom/adapter.py:39` | _dry_theta_view + _temperature_from_theta(dry) + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 24 | `wsm7_adapter` — `src/gpuwrf/coupling/scan_adapters.py:331` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 26 | `wdm7_adapter` — `src/gpuwrf/coupling/scan_adapters.py:524` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 28 | `thompson_aero_adapter` — `src/gpuwrf/coupling/physics_couplers.py:2118` | _dry_temperature_from_state -> T/pi -> _theta_m_from_microphysics | HANDOFF_PRESENT |
| 30 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 32 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 38 | `thompson_hail_adapter` — `src/gpuwrf/coupling/thompson_hail_adapter.py:5` | _dry_theta_view + T=th*pi + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 40 | `morrison_aero_adapter` — `src/gpuwrf/coupling/scan_adapters.py:418` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 50 | `p3_adapter` — `src/gpuwrf/physics/p3/adapter.py:22` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 51 | `p3_adapter` — `src/gpuwrf/physics/p3/adapter.py:22` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 52 | `p3_adapter` — `src/gpuwrf/physics/p3/adapter.py:22` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 53 | `p3_adapter` — `src/gpuwrf/physics/p3/adapter.py:22` | _dry_theta_view + _theta_m_from_microphysics | HANDOFF_PRESENT |
| 54 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 55 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 56 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 95 | `not bound on this main snapshot` — `src/gpuwrf/contracts/physics_registry.py:93; src/gpuwrf/coupling/physics_dispatch.py:149` | N/A | UNBOUND |
| 97 | `goddard_adapter` — `src/gpuwrf/coupling/scan_adapters.py:303` | NONE; state.theta -> dry-TH kernel; direct replacement | MISSING_PREP_AND_FINISH |

T/pi/qv checks:
- MP0: No microphysics handoff.
- MP1: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP2: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP3: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP4: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP5: Dry TH/pi; vapor mixing ratio enters prepare_level, which uses q=vapor/(1+vapor), then finish_level returns q/(1-q).
- MP6: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP7: Dry TH, total pressure Pa, Exner, mixing-ratio qv; WRF phy_prep moist density.
- MP8: Input T is dry theta*pi; pressure Pa; qv mixing ratio. Thompson-specific dry density is intentional.
- MP9: Explicit rounded REAL prep and T=theta*pi; returns dry T/pi before moist finish; no helper-name false positive.
- MP10: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP11: No operational State caller to audit; no implementation/fidelity claim.
- MP13: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP14: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.
- MP15: Dry TH/pi; vapor mixing ratio enters prepare_level, which uses q=vapor/(1+vapor), then finish_level returns q/(1-q).
- MP16: Dry TH; PII from total p; qv mixing ratio. FP64 island where registered; precision is not certified by this source audit.
- MP17: 17 normalizes to18; 19/21 select one-moment variants; qv mixing ratio, total-density multiplier (1+qv) explicit.
- MP18: 17 normalizes to18; 19/21 select one-moment variants; qv mixing ratio, total-density multiplier (1+qv) explicit.
- MP19: 17 normalizes to18; 19/21 select one-moment variants; qv mixing ratio, total-density multiplier (1+qv) explicit.
- MP20: No operational State caller to audit; no implementation/fidelity claim.
- MP21: 17 normalizes to18; 19/21 select one-moment variants; qv mixing ratio, total-density multiplier (1+qv) explicit.
- MP22: 17 normalizes to18; 19/21 select one-moment variants; qv mixing ratio, total-density multiplier (1+qv) explicit.
- MP24: Dry TH; PII from total p; qv mixing ratio. FP64 island where registered; precision is not certified by this source audit.
- MP26: Dry TH; PII from total p; qv mixing ratio. FP64 island where registered; precision is not certified by this source audit.
- MP28: Input T is dry theta*pi; pressure Pa; qv mixing ratio. Thompson-specific dry density is intentional.
- MP30: No operational State caller to audit; no implementation/fidelity claim.
- MP32: No operational State caller to audit; no implementation/fidelity claim.
- MP38: Dry T; total pressure Pa; mixing-ratio vapor; Thompson-specific dry rho.
- MP40: Dry TH; PII from total p; qv mixing ratio. FP64 island where registered; precision is not certified by this source audit.
- MP50: Dry current TH and dry TH_OLD history; kernel T=TH*pi, P inPa, qv mixing ratio. Source audit does not qualify compiled math.
- MP51: Dry current TH and dry TH_OLD history; kernel T=TH*pi, P inPa, qv mixing ratio. Source audit does not qualify compiled math.
- MP52: Dry current TH and dry TH_OLD history; kernel T=TH*pi, P inPa, qv mixing ratio. Source audit does not qualify compiled math.
- MP53: Dry current TH and dry TH_OLD history; kernel T=TH*pi, P inPa, qv mixing ratio. Source audit does not qualify compiled math.
- MP54: No operational State caller to audit; no implementation/fidelity claim.
- MP55: No operational State caller to audit; no implementation/fidelity claim.
- MP56: No operational State caller to audit; no implementation/fidelity claim.
- MP95: No operational State caller to audit; no implementation/fidelity claim.
- MP97: Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.

The fp64 island only changes dtype; it cannot turn moist theta into dry theta. WRF baseline: module_big_step_utilities_em.F:4834 (dry prep), :5735–5737 (moist finish). Existing independent literal REAL4 values are in proofs/t_mprain/thermo_oracle.json.

Additional suspects (not fixed):
- Generic _rho_from_state uses Thompson dry density (physics_couplers:1579/1587, thompson_column:447/450); legacy rho-argument schemes should be checked against WRF phy_prep rho=(1+qv)/ALT (:4856). This includes helpers-present WDM16/WSM24/WDM26; separate from proven missing theta boundaries.
- Goddard97 goddard_run hardcodes zsfc=0 at microphysics_goddard:852-853, analogous to independently proven Lin terrain bug; no change here.
