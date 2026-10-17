# MERGE — t-thetafix temperature handoff

Reviewed candidate `7c1d87146a81a72a920ca0de7d07a87dad339b13`, source tree
`e7edfa50c36bd0cbf9483104f3aed4746b81cb29`, against the Lin prerequisite `1612acc49`.
The eight MP callers (1/3/4/6/10/13/14/97) correctly prepare dry theta and restore
moist theta through the established WRF-order helpers. No source fix is requested.
This accepts the temperature handoff implementation; full affected forecast
qualification and WDM5's canonical number failure remain open.

Independent evidence [M, CPU]:

- **32/32 caller tests pass**, 16.76 s: two consecutive real Swiss calls through
  direct/production dispatcher seams, all eight actual JAX solvers, finite carry,
  active inputs, thermal/mass/precipitation comparisons to original WRF.
- **16 genuine deletion-mutant assertion failures, zero setup errors**, 2.54 s.
  Immutable mutant `0e55d2494326e5770e7c6e4cd645920a4e9994ef` changes exactly the
  eight dry-theta kernel arguments to moist theta; test sources are unchanged.
  Every failure is the numeric dry-TH input assertion. Producer TF10's 16
  assertion failures and seven final receipt hashes were also independently audited.
- Rebuilt both pristine WRF REAL4 and promoted REAL8 callers privately from
  unchanged original bodies, then regenerated all eight schemes, two steps,
  six observed Swiss columns, 44 levels. **32 arrays per kind are byte exact**
  to the tracked fixtures. Eighteen source hashes, caller bytes and CPU-WRF input
  hash match each manifest. Builds have been deleted.
- Full default native-facade jaxpr identities reproduced:
  Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
  These are CPU trace identities, not GPU execution/readiness measurements.
- Independent Git/AST audit: **only the intended eight functions change**;
  47 other functions, including Lin, WDM6, replacement handling and the fp64
  dispatcher wrapper, are unchanged. Every other `src/` file is byte identical
  to the Lin prerequisite. Producer's 335 shared passes are reused source-bound
  evidence, not represented as independently rerun tests.

WRF wiring and scope: reviewed `module_microphysics_driver.F`'s Kessler, WSM3/5/6,
Morrison, SBU-YLin, WDM5 and Goddard calls, and `module_big_step_utilities_em.F`'s
dry prep / moist finish. Each selected adapter passes dry TH, total pressure in
Pa, Exner and dry-air vapor mixing ratio, then uses OLD dry theta/vapor and NEW
dry theta/vapor in the original three-term moist finish. No additional floor,
repair, persistent State field, cache provider or altered precipitation/number
mapping is introduced. The fixture intentionally holds port geometry/Exner and
density conventions equal: it isolates the temperature seam and does not prove
full WRF geometry/density/terrain parity. The existing non-release fp64 islands
are preserved, rather than newly qualified as WRF REAL arithmetic.

**WDM5 canonical number gate remains FAIL.** Independently regenerated original
REAL4 versus REAL8-with-REAL4-carry itself differs by **98,852,928 #/kg** in Nn/Nc
at step 1, column 1, level 10 (zero-based). The actual final dispatcher reproduces
the producer's result: **4/24 Nn/Nc column-field steps fail** the canonical
2%/100 band, Nr passes 12/12; all 36 number column-field steps pass against
promoted pristine reference. Thus the kind-sensitivity explanation is measured
with pristine bodies. It is not a canonical pass, tolerance change, waiver, or
proof of acceptable forecast drift. These WDM5-specific Nn/Nc initial values
are disclosed initializations, not observations. Graupel is only a trace;
new active graupel qualification is not claimed.

Performance/memory: measured default program identity supports unchanged default
arithmetic and memory schema. For affected options, seven old density paths
already calculate dry theta, so reuse by compilation is possible [I]; the new
moist finish adds elementwise thermal arithmetic. There are no new loop passes,
host transfers, host synchronizations, persistent buffers or baked assets in
the source change. Actual GPU launches, fusion/layout, peak temporary memory
and time are **[U]** under this CPU-only critic contract. The scheduled composed
GPU runs must assess them; no GPU speed or efficiency result is asserted.

Required next work: integrate after Lin, then rerun affected composites
**c02/c04/c05/c06/c08/c09/c10/c25/c61**, retaining WDM5 number/D6 failure status
until the manager's scientific disposition. Separate raw-theta callers in
`_surface_layer_bottom_inputs`, `_flux_handles_from_hfx_qfx`, Tiedtke, nTiedtke,
GF, BMJ and `_pbl_surface_forcing` remain source-audit findings needing their own
caller/tendency oracles; they were reported by the producer and are outside
this eight-MP change. No LSM solver is implemented in this file.

Artifacts: `RV01.xml/.log`, `RV02-audit.json/.log`, original build/generate logs
and regenerated fixture manifests, `RV03-numbers.json/.npz/.log`,
`RV04-identities.json` plus original identity receipts, `RV05-mutant.xml/.json/.log`,
and `CLEANUP.json`, all under `<USER_HOME>/wrf_gpu2_lanes/rv-t-thetafix/`.
Own detached snapshot, original builds and pytest temporary directory are absent;
no disk JAX cache was enabled. No GPU was used.
