# SAS family cumulus support

**Status in v0.4.0:** `cu_physics=4` is not supported by `gpuwrf run` (Python API only). Options 14, 94, 95, 96 and 99 are **validated 24 h**: each passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Their compiled GPU code exceeds the project's code-quality limits for speed and robustness; they run correctly, but their speed has not been measured.

WRF 4.7.1 Registry numbers refer to different source modules, even when names are similar.

| `cu_physics` | WRF source | Current support |
|---|---|---|
| 4 | `module_cu_scalesas.F` | Python API only, checked against the WRF module. `gpuwrf run` refuses it because original WRF ARW makes this selection fatal; WRF suggests 95. |
| 14 | `module_cu_ksas.F` | Native REAL32 distinct port, scalar and momentum held rates. CPU check with a documented compiler-contraction deviation. Validated 24 h. |
| 94 | `module_cu_sas.F` | CPU check against original WRF. Validated 24 h. Deep `SASCNVN` plus ARW `OLD_ARW_SHALCV`. |
| 95 | `module_cu_osas.F` | Distinct old OSAS with recorded-draw CPU check and deterministic restart-safe random numbers. Validated 24 h. |
| 96 | `module_cu_nsas.F` | Native REAL32 distinct deep and own shallow port, scalar and momentum held rates. CPU check with a documented conditioning deviation. Validated 24 h. |
| 99 | `module_cu_kf.F` | Distinct native old KF, analytic thermodynamics and gradual freezing. CPU check with a documented compiler-conditioning deviation. Validated 24 h. |

cu=94 uses the WRF REAL4 driver and DOUBLE `kind_phys` internal calculations.
Its default shallow convection is included. `sas_pgcon` is bound per domain in both
pipelines; the WRF default is 0.55. `cudt` determines STEPCU, and the scheme refreshes
at step 1 or a step divisible by STEPCU. Held rates are applied and convective rain
accumulated every step. SAS does not add momentum tendencies in the ARW caller.

The seven existing `cumulus_tendencies` restart slots retain REAL temperature,
vapor, cloud water, rain (zero), cloud ice, snow (zero), and precipitation rates.
No new prognostic fields are introduced. The adapter uses the existing increment
application step.

Column evidence and reproducible builds are under `proofs/v040/sas_family`.
The original 455-column input set includes active and inactive soundings; the
deep and default shallow calls are checked independently at DT=54/18 s and
STEPCU=5/1. REAL4 output checks require at most one float32 ULP and at least 99.9%
byte identity per field, exact cloud levels, finite results, and no exclusions.
The promoted REAL8 build is a separate diagnostic, not the production precision.

GPU kernel counts, device memory and forecast speed for cu=94 have not been
measured. The JAX column program includes vertical recurrences and constant
GFS lookup tables. It is opt-in; the released KF path is unchanged.

In the initial five constructed columns, cu=14 literal REAL4/AVX rain, T/QV and winds are bit-identical to the original WRF; QC/QI differ by less than one ULP. REAL8 maximum scaled error is 6.94e-14. Default-ISA compiled rain differs by 3.75509e-6 absolute; pristine Fortran native-FMA control itself moves rain by 3.33786e-6, leaving an excess of 4.17233e-7. A few tiny QC/QI/QV cells also exceed that spread. This is a documented compiler-contraction deviation; tolerance files and product instruction-set settings are unchanged. The 24-hour forecast comparison is the deciding fidelity check.

Cu=96 REAL8 maximum scaled error is 5.70e-13. On the same 400 temperatures observed in the pristine driver, FPVS differs by at most 2 ULP (366 bit-identical). The measured chain is FPVS 2 ULP -> QO 1 ULP -> XAA0 1 ULP (0.00048828125) -> closure XMB 44–68 ULP -> rain 1.043081283569e-6 absolute / 5.798721e-6 relative. AA1 matches exactly. This is a documented conditioning deviation; tolerance files and product instruction-set settings are unchanged. The 24-hour forecast comparison is the deciding check.

Cu=14/96 use the original one-based first-step/STEPCU cadence and retain nine REAL32 rate leaves (the seven existing scalar/precipitation leaves followed by RU/RV) in both checkpoint formats. Malformed, missing and nonfinite active SAS held rates are refused at restart. Cu=14 reads the pre-PBL HPBL_HOLD and current HPBL. Cu=96 reads current HFX/QFX/HPBL and shares pressure-search limits over each original i-row.

The expanded 455-column corpus (44 levels) exercises 31 active KSAS columns and 259 active NSAS columns. Literal REAL8 matches all ten fields at the fixed 1e-9 scaled limit. Native REAL32 errors exceed the original fixed driver limits in some cells; these are documented conditioning deviations, not a claim that the original limits passed. KSAS rain max absolute difference is 1.317262649536e-5 versus pristine WRF native-FMA spread 6.193667650223e-5; its largest same-cell excess over that spread is 1.031160354614e-5 in two rain cells. NSAS rain max absolute difference is 4.450324922800e-5 versus original FMA spread 4.640389233828e-4; largest same-cell excess is 2.554617822170e-6 in 13 rain cells. Original native-FMA NSAS also changes the exact-zero rain/cloud-level branch in column 133, with the same HBOT/HTOP change as the port. The full per-field and per-cell excess accounting is kept in the evidence, without changing tolerance files or product instruction-set flags.

Cu=95 checks the native REAL4 path against recorded original XKT2: all eight outputs are bit-identical across 455 columns at DT54/STEPCU5 (198 active) and DT18/STEPCU1 (196 active). The shared ARW SHALCV, MSTADBT3 and TRIDI2T3 source is identical to cu=94 after entry renaming and Fortran case/comments/whitespace only. The old deep OSAS kernel is separate, including its stochastic top selection, entrainment, downdraft and closure.

**WRF draws from the system clock; gpuwrf is deterministic by design.** The experimental cu=95 port uses Threefry2x32-20 with key (fixed seed 950040, domain identifier) and counter (one-based own step, global row-major column). It converts 53 bits to DOUBLE XKT2 in [0,1). Both CLI pipelines bind the domain id, and both words are saved in every restart format alongside the existing clock and seven held rates. Missing or wrong keys are refused at restart. API callers with multiple domains must explicitly supply distinct `osas_domain_id` values. No additional seed namelist option is introduced.

The promoted-REAL8 diagnostic has a documented exact-zero sensitivity in fully evaporated rain around 1e-20. At DT54/STEPCU5 column 237, this can switch the deep/shallow branch: max RTH difference 2.143312553719e-4 versus pristine original FMA field spread 3.500857767053e-4. The largest rain differences are 3.747002708110e-16 versus original FMA spread 1.400407567687e-13 (54/5), and 2.066402604584e-14 versus 4.668834763244e-14 (18/1). These are kept as deviations; the original fixed REAL8 relative/level limits are not counted as passed. The original replay reproduces the complete original WRF stdout byte for byte with the same recorded draws. The native REAL4 primary path remains bit-identical.

Cu=14 and cu=96 each completed a short CPU run on a real Swiss 42×42×44 case of three 18-second steps with Thompson/MYNN/Noah-MP and radiation disabled: zero nonfinite values and all 66 guard counters zero. These are short coupling checks; the 24-hour forecast comparison is separate. Full-radiation CPU arms timed out at 900 seconds during compilation.

Cu=99 is a distinct KFCPS/KFPARA port. All ten promoted-REAL8 output fields pass the fixed per-column 1e-9 limit on the five original soundings (two active, three inactive); source budget failures and moisture-borrowing events are zero. The literal native control passes eight of ten fields; tiny QI/QS rates exceed their frozen relative bounds. The default-ISA compiled control passes thermal rates/lifetimes but has documented conditioning deviations: rain max absolute difference 1.937150955200e-6 versus pristine Fortran native-FMA spread 1.892447471619e-6 (excess 4.470348358154e-8); QI 6.664890861430e-12 versus 4.575895218295e-12; QV 8.469669410260e-11 versus 7.844391802792e-11. Full per-field and same-cell excesses are retained in `proofs/v040/sas_family/kf99_conditioning.json`. These original strict native checks are not counted as passed; no tolerance or product instruction-set change was made, and the 24-hour forecast comparison is the deciding check.

Old KF uses the original pressure-weighted dry-theta source mixture, fixed 1500-m cloud radius, analytic liquid/ice TPMIX/TPDD and gradual DTFRZNEW freezing. Remaining CAPE includes the overshoot through LTOP-1. Source W0AVG and NCA are REAL32 restart leaves, alongside seven held REAL32 rates. KFCPS updates W0AVG on each scheduled call and skips triggering only for NCA greater than half a timestep: equality runs. Old advance_ppt clears six scalar rates only when NINT(NCA/DT)<=0, then decrements positive NCA. This differs from KF-eta's <=1 rule; PRATEC is retained in either case. Scalar application follows the original add_a2a boundary ring; precipitation and countdown use the full advance_ppt range.

The old-KF component CPU measurements (five 40-level columns, contended single core) are: default-ISA native compile 7.603 s, warm median 2.839 ms, temporary memory 88,600 bytes; literal native control compile 21.867 s / warm 10.509 ms / temp 90,792 bytes. These are CPU component measurements, not GPU launch, roofline or forecast speed claims. GPU cost has not been measured. The retained direct step-entry coupling remains an integration caveat.

The W0AVG recurrence has a separately documented CPU code-generation effect. On the original five native columns, the AVX literal control is bit-identical for normal values but flushes 15 subnormal upper-level values (max 6.983258793086e-40). The default CPU control differs by at most 2.980232238770e-8 m/s, equal to the maximum original Fortran FMA spread; the registered one-ULP check fails and is kept in the evidence. No product ISA or arithmetic change is made. The CPS unit check additionally verifies an independently derived forward-error envelope: for nonadaptive STEP=5, define S=(9|W0_old|+0.5(|W_lower|+|W_upper|))/10, u=eps32/2 and gamma6=6u/(1-6u). Each implementation has at most six elementary operations, so their rounding difference is bounded by 2 gamma6 S. Input and intermediate flush-to-zero contribute at most five minimum-normal REAL32 values for STEPCU>=1. The check uses 2 gamma6 S + 5 tiny32, entirely from original inputs; this is not a forecast tolerance change.


All five supported family options (14/94/95/96/99) completed three-step CPU coupling checks on a real Swiss case: zero nonfinite values and all 66 guard counters zero. The 94 run includes RRTMG; the other four use radiation disabled to keep compilation bounded. `proofs/v040/sas_family/cpu_coupled.json` records the complete resolved work. These checks establish coupling mechanics; the 24-hour forecast comparison is separate.

The final native precision audit finds zero double-precision (float64) instructions in the optimized compiled code for 14/96/99. Typing exact zero/one selectors removes 63/73 accidental float64 storage/select instructions from 14/96; every native output stays bit-identical to the previous version. CPU temporary memory changes from 5,684,864 to 5,448,256 bytes (14) and 6,749,432 to 6,662,072 bytes (96); 99 remains 88,600 bytes on five columns. These matched-input CPU measurements do not establish GPU memory, kernel launches or speed. GFS94/95 keep the DOUBLE islands present in original WRF. The default forecast program is unchanged.


A caller correction applies WRF `add_a2a` to the scalar consumers of 14/94/95/96: specified and nested grids exclude the outer mass ring. Native momentum also excludes the outer j rows of U and the outer i columns of V. Periodic-x restores x bounds only; it does not restore specified/nested j bounds. Raw held rates and precipitation retain their full source ranges. Original unchanged Fortran routines provide all four specified/nested/periodic-x witnesses. The retained cu=4 API and existing PBL helper bodies are unchanged.

The numerical checks reject nonfinite candidate or reference arrays, auxiliary operands and derived bounds, including exact levels and lifetimes (`proofs/v040/sas_family/CRITIC_FIXES.json`).

After the caller correction, the four affected three-step CPU runs on a real Swiss case (radiation off, no-repair mode) are finite and every guard counter is zero (22 guard families × 3 event types = 66 counters).

At a 44×70×120 REAL32 CPU shape, the corrected consumer uses zero temporary bytes and a median of 0.459474 ms, versus 0.649125 ms for the previous consumer with the same inputs; compile time is 0.233301 s versus 0.186446 s. A static-slice alternative needed 1,478,400 temporary bytes and 1.147139 ms and was discarded. These are CPU component measurements, not GPU launch, memory or forecast-speed claims.
