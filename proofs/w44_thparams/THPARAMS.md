# Thompson runtime parameters — W1 candidate

Status: Nt_c/TNO/ATO/AV_C have actual GPU component evidence and are the only
names the public binder accepts. Product-root GPU E41/cost and integration
gates remain open. Ice/rain fall laws and autoconversion have separate CPU
evidence and are refused at runtime until their own GPU/E41/cost gates exist.
The combined all-eight CPU default class is measured (see the FIX closure below).

These are **gpuwrf extensions** to WRF 4.7.1. WRF declares these coefficients
as private REAL PARAMETERs, except the two autoconversion controls, which are
source literals. Comparison at changed values requires a private recompiled
WRF module with the corresponding definition changed. The original WRF
source/reference is preserved.

## Candidate ranking and scope

Sensitivity is an inference [I] from each equation, not a measured forecast
sensitivity ranking. Definitions below refer to pristine
`WRF/phys/module_mp_thompson.F`.

| Priority | Candidate | Clouds/rain sensitivity [I] | Cost/dependency | Decision |
|---|---|---|---|---|
| 1 | Nt_c (:88) | Droplet size, autoconversion, freezing, sedimentation and cloud radius | Runtime nu/closures/index; complete WRF gamma/Nc grids are Nt-independent and reusable | CPU and GPU component gates pass; product-root GPU gate pending |
| 2 | ATO (:189), TNO (:188) | Ice-nucleation number, exponential temperature sensitivity (:2625); cold-cloud and precipitation pathways | Pure scalars, no dependent initialization/table rebuild | Implement now |
| 3 | AV_C (:163) | Cloud-droplet fallout, especially low cloud; smaller direct rain effect | Pure scalar in mass/number fall laws (:3661/3663); mp8 carries mass only | Retained candidate under the manager's documented round-off rule; TP14 changed-WRF gate passes |
| 4 | Autoconversion threshold/rate (:2180/:2186) | Warm-rain activation and conversion rate | Pure source-literal scalars; original limiters retained | Separate CPU gates pass; [evidence](AUTOCONVERSION.md) |
| 5 | Rain/snow/ice fall laws | Broad precipitation/collection response | Ice av_i is scalar; rain av_r needs scalar init coefficients; snow av_s changes collision/collection tables | av_i/av_r separately CPU-qualified; snow deferred, no partial av_s substitution |

`ccn_conc` is not a Thompson control. WRF Registry.EM_COMMON:2668 defines it,
but module_microphysics_driver.F:2729/2776/2827/2879 passes it to WDM; Thompson
uses its separate Nt_c PARAMETER. No ccn_conc-to-Nt_c alias is provided.

## Planned gate

- OFF: Swiss b977712f / PROD 321aa651 canonical release identities and raw native HLO.
- ON default: compiled output byte identity or the manager-authorized localized round-off class; changed values with one traced signature/AOT key.
- Original WRF recompiled with changed constants on real-structured cloudy/ice/graupel columns; frozen mixed-phase increment tolerances, finite checks first.
- Asserted GPU, E41 cubin on actual product programs, warm-root A/B and memory/ops/copies. Preparation is split into four bounded 900-second arms sharing one cache; [recovery and measured cap basis](NIGHT_RECOVERY.md).
- Shared registry/binder from w44-params; the assigned cr-w44-thparams critic reviews CPU/source evidence before the GPU window, then the GPU receipts in the same review.

## CPU evidence so far

- TP03 [M CPU, interpreted Pallas inside outer JIT]: TNO/ATO default values
  produce 20/20 byte-identical output leaves on 24 real mixed-phase columns,
  44 levels, dt=18 s. Changed TNO=10 / ATO=.608 alter 6/7 leaves, with exactly
  two traces (OFF and ON); no third trace for changed values.
- Input coverage [M]: qc/qr/qi/qs/qg nonzero cells 198/265/256/360/153.
- TP04 [M WRF CPU]: recompiling the original Thompson module at defaults
  reproduces all 12 frozen reference fields byte-exact. Table SHA256s match
  the original fixture provenance. Original files are read-only.
- TP05/06 [M CPU]: separately recompiled WRF with TNO=10 and ATO=.608 passes
  the frozen mixed-phase 10-field increment gate against native CPU. Keeping
  default parameter values instead fails, proving both controls are active.
- AV_C initially failed the stricter BYTE interpretation. The manager clarified
  that localized few-ulp differences are admitted if the changed-value WRF and
  common-key gates pass. TP02 [M CPU] differs in 2 qc cells
  (max 1.7763568394e-15 kg/kg) and 2 cloudw precipitation cells
  (3.6379788071e-12 mm). TP07 optimized HLO localizes the cause: the static
  expression folds three constants into 8620934144; the runtime operand
  changes multiplication grouping. Both differences are at most **2 WRF REAL
  ulps**. Maximum relative differences are 1.9227593e-7 (qc) and 1.3883139e-7
  (cloudw). Cloudw is computed in float32, then widened into a float64 output:
  its carrier difference is 1,073,741,824 float64 ulps, explicitly not a claim
  of two float64 ulps. The OFF equations and frozen WRF-oracle tolerances did
  not change. TP14 original WRF recompiled at AV_C=63389200 passes the existing
  increment gate, and ignoring the changed value fails. Later GPU evidence
  for the four selected parameters is recorded below.
- TP10 [M CPU CUDA-target lowering]: native OFF raw HLO is BYTE-exact against
  c653483682, 628924 bytes, SHA256 `12a9494b4942675ed224d5778ee5b0de72ad7a968abc700198e242c2fb486e3f`.
- TP12 [M CPU CUDA-target lowering]: TNO/ATO values 5/.304 versus 10/.608
  produce identical native raw HLO, SHA256 `c8fd95496149f2ac9f31a4d140a88881ba37f042cc0cb1a8fecb6c5abe6f5cb4`.
  Later whole-program and four-parameter GPU evidence is recorded below.
- TP15 [M CPU]: 164 tests pass, including common static/aval keys for changed
  values, different keys for different selected names, value digests, original
  OFF adapter object identity and finite-first rejection controls.
- TP16 [M CPU trace]: complete Swiss `b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a`
  and PROD `321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b` identities remain exact.
- TP17 [M CPU CUDA-target lowering]: **whole Swiss OFF raw HLO is BYTE-exact**
  for 6fab31f571 versus c653483682: 22468110 bytes, 1080 native custom calls,
  SHA256 `3641e8bed499c14bfab5cde473e65b4283f2a6d26d93102446a2b7439d04c95e`.
  Receipt `TP17/WHOLE_OFF_IDENTITY.json`, SHA256
  `5170a5cb1c4468773fdbda3070fee4bae998f5821bb33cbf3d13b5609d2ab30b`.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/w44-thparams/TP02` through `TP07`.
Disposable WRF objects/executables removed after the reference receipts.

## Nt_c qualification (ALISIOS priority, scope expanded 2026-10-23)

- NT01 [M original WRF]: compact `thompson-ntc-v1.npz` is 2392570 bytes,
  SHA256 `997ba2362c622afd555adcbfcaaeffbd1f3f2aea21e4288e077c45f611705305`.
  Both default Nc=100e6 freezing planes are BYTE-exact to the existing asset.
  Fresh original Fortran reproduces all 75 gamma and 30 reciprocal REAL bits,
  two nu=12 compound prefixes, and 54 shape/index boundary controls.
- Original mu_c is diagnostic-only; numerical nu_c is
  `MIN(15,NINT(1000e6/nc)+2)`. The complete gamma/Nc grids are Nt-independent.
  Fixed mp8 ice-nucleus index is 28. The new table adds 2637360 logical bytes
  over the former two fixed-Nc planes and is used only when Nt is enabled.
- Nt scalar preparation runs before Pallas, including the original DOUBLE
  logarithm. Seven REAL operands carry nu/index/closures into the column
  kernel. Host caching contains only NumPy arrays (E206).
- NT03/NT08 [M CPU]: Nt=50/150/200 cm⁻³ each passes the original frozen
  24-column, 10-field WRF increment gate; all omission controls are killed.
  Each marine value changes 16 output leaves, with only OFF and ON traces.
- NT07 [M CPU]: all seven default closure bits and all 54 original indices
  match. Operand isolation finds rounding only in Xdc (20 cells, max
  1.9073486e-6 µm) and MVD (15 cells, max 1.8189894e-12 m); cloud density,
  lambda and activity stay exact. The default class extends only Nr to
  **3 REAL ulps**, two cells, maxabs 3.0517578e-5 kg⁻¹ and maxrel 2.6036019e-7.
  Other fields keep two ulps; the WRF-oracle tolerances are unchanged.
- NT-RADIUS50/100/150 [M original WRF vs CPU]: cloud/ice radii BYTE; snow
  maxrel 9.7325953e-6 < preregistered 1e-5 (Nt-independent inherited error).
  Default dynamic/static producer is 3/3 BYTE; one JIT handles all values;
  omission kills 178/177 cloud cells. No flux or forecast claim.
- NT09 [M CPU]: the original inactive-cloud MVD predicate (:2168–2176) is
  restored only for Nt. All 80 output leaves across OFF/default/50/150 arms
  remain BYTE-exact to NT08, so the original WRF gates carry forward.
- NT10 [M CPU]: **200 tests pass**, including original-reference radius
  copyback, one producer call, held values into both LW/SW, omission and
  nonfinite controls, keys/digests and the inactive-cloud predicate.
- NT11 [M CPU CUDA-target lowering]: final Nt OFF raw HLO is BYTE to the
  pristine lane baseline (22468110 bytes, SHA3641e8be). ON default/changed
  values produce identical raw HLO (22473664 bytes, SHAe6f44c33), canonical
  graph and static/aval keys. Both have 1080 native call sites; this count is
  not a measured device-operation census. No GPU compilation/execution claim.

The Nt adapter suppresses the fixed-Nt radius block locally, invokes the
unchanged original radius equation code with a private traced Nt binding,
then restores the existing radius leaves. It performs no duplicate producer
work and changes no shared radiation/coupler source.

## Shared dependencies and integration

The family mechanism is imported unchanged from w44-params commit d28bb019ee
(runtime_parameters.py blob 9c4161528ecf842c90f5ca549bbe0f15d4350702), then extended
with one Thompson family. Catalog/restart-helper imports come from 8a443ffeaf.
Manager granted the namelist/registry/driver seams; patches were posted to
MAIN MAILBOX before their commits. w44-params reviews the shared seam.

The existing adapter object is returned when OFF. ON selects the new private
adapter, which reuses the original column preparation/copyback and passes the
REAL operands into the existing Pallas kernel. It adds no State leaves.
Named values are children; sorted selected names are static aux to prevent
same-size/different-name key collisions. The registry's restart digests and
the branch owner's static/digest family lists include Thompson.

The shared CLI/store integration remains a merge dependency on w44-params and
w44-branch. This isolated lane did not copy their CLI or store implementations.
Checkpoint helpers are tested here; whole checkpoint/resume qualification is
not claimed from those helper tests.

## Cost and pending gates

OFF canonical/native raw identity is measured; no numerical revalidation of
that exact program is needed. ON adds scalar operands with unchanged output
shapes. Nt adds a small scalar pack and 2.64 MB of logical table data.
The actual GPU component reports temporary storage of 3647272 B OFF and
3647516 B ON (+244 B); argument storage rises from 63360 B to 63376 B.
Generated-code sizes are 110393168/113041960 B. These are component executable
statistics, not whole-forecast memory. Product-root copies, cost and E41
remain open; unchanged complexity is not performance evidence.

The immutable v3 launcher has separate per-program cold preparation, CPU E41,
warm-root and profile phases. Its CPU launch smoke and actual PROD input/key
checks pass. The manager moved the GPU window to 25 October after the power
failure; no worker self-launch is authorized. Real
PROD roots provide regression/cost evidence; the original WRF column oracles
provide the physics evidence. If a root has no active sensitivity, the receipt
records that and the nonvacuous cloudy/ice/graupel component gate remains mandatory.

## Actual GPU component and separate CPU follow-ups

- GF01/GF02 [M GPU outputs, CPU scoring]: source `2e66785c50`, real 24-column
  corpus, Nt_c/TNO/ATO/AV_C selected. ON default is 20/20 BYTE; Nt=50/150e6,
  TNO=10, ATO=.608 and AV_C=63389200 pass all five recompiled-WRF checks and
  all five default-value omission controls fail. Seven event probes preserve
  prognostics BYTE and report zero input-nonfinite, pre-mask-nonfinite and
  inadmissible counters.
- Four component/event executables pass E41. The failed root executable was
  the harness-specific `fused_d01`, selected by omitting the product history
  observer. Its 664-instruction fusion failure remains a retained finding.
  The manager requires normal E201/E41 on the corrected product `d01`/`d02`
  programs; no release failure is waived or inferred from another program.
- [Autoconversion](AUTOCONVERSION.md): source-literal threshold and coefficient,
  CPU default 20/20 BYTE, two changed original-WRF gates and omission controls,
  native value-independent HLO, 97 focused tests.
- [Ice fall law](AV_I.md): changed original-WRF gate and omission control,
  native OFF identity and value-independent ON HLO; isolated default class
  reaches four REAL ulps, with one widened-REAL ice-precipitation difference.
- [Rain fall law](AV_R.md): all dependent scalar initializers included; changed
  original-WRF gate and omission control, OFF/native-key evidence, 94 tests;
  isolated default rain-number difference reaches six REAL ulps.

These isolated classes do not qualify a combined-eight-parameter program.
The public binder refuses av_i, av_r and both autoconversion controls until
their own GPU/E41/cost evidence exists; low-level CPU qualification is retained.
GPU-root AOT reuse, performance/memory/ops, critic acceptance and merge remain
open. Component receipts: `gpu/component/prime/receipt.json` and
`gpu/scoring/prime/SUMMARY.json` under the lane artifact directory.

## Critic phase-A FIX closure (cr-w44-thparams verdict 85fb1868, Opus takeover 2026-10-24)

Repaired head = lane/w44-thparams rebased on lane/w44-params `aa923ff4fa`; own
delta numstat identical before/after the rebase. All results [M CPU] unless noted.

1. Adapter seam: `selected_adapter(namelist, default_adapter)` returns the
   caller's module-level `thompson_adapter` when OFF, so the B32/B35 ordering
   tests intercept again. Consumer files: see the TK03 receipt below.
2. Docs/catalog: all eight names documented with their per-name default
   classes; the public binder and runtime adapter accept only nt_c/tno/ato/av_c.
3. Changed-value WRF gates committed: 9 recompiled pristine-WRF references
   (`tests/w44_thparams/fixtures`, SHA-pinned manifest) on the 24 frozen
   mixed-phase columns; `tests/w44_thparams/test_changed_wrf.py` **11/11 pass**
   (9 changed values pass and their default outputs fail the gate; TNO and AV_I
   kernel hardcode mutants raise the gate AssertionError).
4. Combined default class, all eight at WRF defaults vs OFF
   (`receipts/combined_defaults.json`): qc 2 (4 cells), qr 3 (12), qi 3 (22),
   Ni 4 (17), Nr 6 (22), Ng 1 (2) REAL ulp; widened cloud-water and ice
   precipitation carriers 2 and 1 REAL ulp; qv, qs, qg, Ns, T and rain, snow and
   graupel precipitation BYTE. This is the union of the isolated classes. Cause:
   replacing only the eight traced constants and their derived REAL prefixes by
   the static values restores **20/20 leaves BYTE**. The lane's older generic
   round-off helper (limit 2–3 ulp) still reports qr/qi/Ni/Nr above its limit;
   that helper is not the registered gate, item 5 is.
5. E100 floor (`receipts/e100_combined.json`): the default pristine build
   reproduces the frozen fixture 12/12 BYTE. Over 25 seeded members, ±1 REAL ulp
   on every nonzero qv..th input, the pristine output spread reaches
   qv 9.3e-9, qc 8.8e-9, qr 5.8e-10, qi 4.7e-10, Ni 25.8, Nr 7.3e-3 and th
   9.2e-5. **Every field of the combined default class lies within its field
   floor** (Ni 2.4e-4 vs 25.8; Nr 1.5e-3 vs 7.3e-3; qr 1.2e-10 vs 5.8e-10).
   Disclosed: 1 Ni and 3 Nr cells lie above the stricter cellwise envelope.
6. Admission: tno/ato/av_c/av_i/av_r only within 0.5–2× the WRF default
   (REAL32 comparison, inclusive); 1e9 and the neighbouring REAL values are
   refused (pytest).
7. Integration: rebased on the owner registry; `checkpoint_branch.PARAMETER_KEYS`
   includes thompson_parameters (branch test); restart family tuples keep it.
   The Nt_c asset reader is registered (4 sites, binding
   `thompson_cloud_number_asset`). `baked_table_hash` keys
   `thompson_nt_c.host_tables` only when nt_c is selected; a collision test
   checks that a one-ulp table change splits that key, while OFF and tno keys
   stay unchanged.
   Consequence: the ON nt_c cheap key on this head differs from a2df's
   prepared keys; the native program does not.
8. GPU scope: av_i, av_r and both autoconversion names are refused by the
   public binder/adapter (option b) until their GPU/E41/cost evidence exists.

Carry-over of the frozen Nt4 GPU packet (`receipts/native_hlo_a2df_vs_head.json`):
native Thompson raw HLO through one alias path, frozen a2df vs repaired head,
OFF `12a9494b…` (628924 B) and Nt4-ON `97e688e6…` (638026 B) are **byte-identical**,
canonical jaxprs equal. E225 launcher check: `receipts/LAUNCHER_VALIDATION_v3.json`.

## Release-note text (for integration by w44-docs)

The optional Thompson runtime family exposes Nt_c, TNO, ATO and AV_C for native
mp_physics=8 experiments. They are WRF compile-time PARAMETERs, exposed as a
gpuwrf extension; CPU-WRF comparisons require recompilation at the same values.
The family-off default program is unchanged. Enabling default AV_C can change
cloud water and its sedimentation diagnostic by up to two WRF REAL ulps on
the qualified real-column corpus because a traced coefficient changes constant
folding. The float64 cloud-water precipitation carrier contains a widened
float32 result; its float64 ulp count is disclosed separately. The changed-value
pristine-WRF checks retain their original tolerances. Nt_c supports the marine
50–150 cm⁻³ use case, including dependent cloud moments, freezing lookup and
held radiation radius; its measured default class adds at most three REAL
ulps in rain number, with all other changed fields within two. The measured
four-parameter GPU component is BYTE at defaults; product-root qualification
is pending. TNO/ATO and the isolated autoconversion defaults are BYTE on the
CPU corpus. Experimental AV_I has qi ≤3 / Ni ≤4 REAL ulps and a one-REAL-ulp
ice-precipitation result widened to float64. Experimental AV_R has qr ≤3 /
Nr ≤6 REAL ulps. The public binder refuses AV_I, AV_R and both autoconversion
controls until their GPU/E41/cost gates pass; their CPU evidence does not
extend the four-parameter GPU claim. The five fall/nucleation coefficients
are admitted only within 0.5–2 times their WRF defaults, a documented research
envelope rather than WRF-declared limits. With all eight names enabled at
their defaults the CPU differences are the union of the isolated classes
(at most six REAL ulps, in rain number); each field's largest difference lies
within the largest change pristine WRF shows under one-REAL-ulp input
perturbations (per-field bound; 1 Ni and 3 Nr cells exceed the cellwise envelope).
