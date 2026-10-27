# w44-params gate record

Candidate status: **CPU gates and launcher preflight pass; GPU and composition
gates remain open**. Not ready for release or the final kernel critic. Source
and receipts are local; no GPU performance or coupled-fidelity claim is made here.

| Family | Implemented | Evidence | Cost / decision |
|---|---|---|---|
| Surface | Restart digest includes actual CTOPO; CLI checkpoint refusal removed after composing branch-owned authentication | PA02 7 CPU checks; composed MC01 33 checks include actual pickle/native-store/CLI consumers and declared-value identity | No model arithmetic/input-layout change in this patch. Host hashing at restart boundaries is additional work; GPU work and final branch acceptance remain explicit |
| Diffusion/damping | Seven optional REAL32 scalars, including `c_s`; value-independent program key and strict active-consumer checks | PB05 27 checks; original WRF sixth-order factor 0.12/0.24; CS01 14 checks include original WRF `c_s=0.25/0.5`, default compiled BYTE and one JIT for two values | Seven selected scalars add 28 input bytes; full device memory/traffic/ops/time unknown. GPU/E41/cost pending; shelve on demonstrated fusion/E41 or speed loss |
| MYNN closure | SQFAC plus selected ALP1, unstable/stable phi and entrainment operands; ALP1 includes first-call initialization | W301/W302 14 checks; original-WRF changed-value bounds; W303/W304 native and full Swiss/PROD OFF exact; W305 all-five default BYTE in both regimes; W306 value-independent native program | All five selected scalars add 20 input bytes; full GPU memory/traffic/time remain unmeasured. GPU/E41/cost gates pending; budget merges first, then params rebases |

The manager accepted explicit `khdif`/`kvdif` refusal on 2026-10-23. They require
constant-K `km_opt=1`, which the native driver does not support. No tuning claim.

| Knob | WRF consumer / configuration | Runtime status |
|---|---|---|
| `c_s` | 2-D Smagorinsky, `diff_opt=1, km_opt=4` (PROD/WN3) | Added; CPU gate green, GPU gate pending |
| `c_s` | 3-D Smagorinsky `km_opt=3`; SMS horizontal blend `km_opt=5` | Static consumers exist; runtime extension deferred and refused |
| `c_k` | Supported `diff_opt=2, km_opt=2/5` LES viscosity and dissipation | Runtime deferred/refused: `Les3dConfig` casts plus REAL `ce1/ce2` handling and separate LES program qualification needed |
| `khdif`, `kvdif` | Constant-K `km_opt=1` | Unsupported configuration; refused |
| `diff_6th_factor` | Enabled sixth-order filter, independent of the K closure | Added for specified/nested native paths; LES TKE and passive-tracer consumers refused |
| `dampcoef`, `zdamp` | Native `damp_opt=3`, independent of `km_opt` | Added |
| `w_alpha`, `w_beta`, `w_crit_cfl` | WRF stage `w_damping=1`, independent of `km_opt` | Added |

## MYNN subset selection

ALISIOS clarified W3 on 2026-10-23: mixing-length coefficients and upper bound,
stability/closure coefficients, and EDMF entrainment rate. SQFAC is the first
implemented member; it does not by itself cover that expanded request.

The preregistered selection probe uses the real 12 daytime and 12 nighttime
44-level columns from `mynn_predict_saw_wrf.json`, three steps with the original
surface forcing held fixed, and the native MYNN program compiled on CPU.
It measures changes in mixing length, diffusivities, TKE, wind, temperature and
moisture. This is conditional column sensitivity, not forecast skill or a WRF
fidelity comparison. `mynn_sensitivity.py` clears JIT caches for every static
perturbation and requires the entrainment packing and stable-function rebinding
controls to reproduce baseline bytes. **MS01 completed:** ten arms in 188.29 s
on CPU; both unchanged-value controls passed byte equality. The selected
coefficients are implemented in `fd4a66434f`; composed-program and GPU gates
remain open.

| Candidate | WRF meaning and reason to examine | Selection status |
|---|---|---|
| `NL_ALP1=0.23` | Option-1 TKE-weighted PBL-depth length integral; directly affects the active mixing-length calculation | SELECT: +10% gives Kh relative RMS change 9.711% day / 1.000% night; −10% gives 11.023% / 1.225% |
| `NL_ELT_MAX=400 m`, water base `350 m` | Upper limit of that integral, including the unchanged water hurricane increment | DEFER: +10% and half caps both give zero changed cells in all ten fields and both regimes; need a real column where the cap acts |
| `CPHM_UNST=16` | Unstable branch of the Puhales momentum stability function feeding surface TKE production | SELECT: +10% gives TKE relative RMS change 0.182626% day; max 0.0035432 m²/s² |
| Stable `am_st=6.1` | Stable branch of the same function; complements the daytime coefficient | SELECT: +10% gives TKE relative RMS change 0.203961% night; max 0.0107135 m²/s²; daytime fields BYTE |
| EDMF numerator `0.33 m/s` | Entrainment numerator divided by bounded plume velocity and width; existing native scalar-stack slot | SELECT: +10% gives cloud-water relative RMS change 3.56081% day and max temperature change 0.00646973 K; nighttime fields BYTE |

Relative RMS change is `RMS(perturbed − baseline) / RMS(baseline)`, over all
12×44 cells at step 3 of each cohort: daytime `dt=18 s, dx=3000 m` (54 s total),
nighttime `dt=6 s, dx=1000 m` (18 s total). These are within-cohort comparisons,
not a matched-duration comparison between regimes. The metric measures difference magnitude, not
the direction of the change. Zero cap sensitivity on these columns does not
establish that the caps never matter in other boundary layers.

The proposal avoids changing the linked B1/A1/A2 and derived closure-constant
bundle: changing one without recomputing its dependents would define an
inconsistent closure. Inactive mixing-length options are also deferred. Final
inclusion requires the same OFF identity,
changed-value pristine WRF, value-independent AOT, and GPU cost gates. No new
MYNN parameter is advertised as release-qualified before those gates pass.
Budget granted the narrow hooks and forwarders at 23:45Z; the exact full diff
`d0cf1822…` was posted before committing. ALP1 reaches the first-call QKE
initializer as well as the ordinary turbulence step. Warm-start initializer
keywords must be preserved when composing that lane's changes. Detailed
selection receipt: `receipts/MS01_SELECTION.json`.

## Critic phase-A FIX status (cr-w44-params VERDICT FIX on 203172dfed; takeover 2026-10-24T21:11Z)

All numbers [M CPU] unless marked; tree = lane/w44-params rebased onto main `6e6e7a522c`.

**Critic verdict 2026-10-25T00:49Z: PRE-OK** (cr-w44-params on 42e980f9b1, covering later heads with identical src/tests/scripts/docs/tools): 17/17 independent source-file mutants killed, suites green in both conventions, own re-compare of ID01 identical, composition with main incl. constfold (merge e532904293): flag 0 -> 1-ulp class, flag 1 -> BYTE. Final MERGE after phase-B GPU receipts or on manager request.

| FIX | Status | Evidence |
|---|---|---|
| 1 caller-deletion survivors M06–M11, M15–M17 | Positive caller tests + one in-process mutant per survivor (live production function re-compiled with the forwarding removed or hard-coded; must fail the positive test with its own assertion) | `tests/w44_params/test_production_callers.py`; FX02 FAST1: 40/41 pass, the w_beta mapping mutant first patched the wrong binding (fixed `cefc083f69`, rerun FX03) |
| 1 MYNN source-leaf branch | Fixture now sets `rad_rk_tendf=1` (production default; the branch is selected by `namelist.rad_rk_tendf`, critic 21:07Z) | 5 `[*-True]` cases pass (FX02) |
| 2a damping class documented | See next section; slot-level test `test_dampmag_slot.py` | 12 pass: traced slot = WRF REAL `f32(dts)*f32(dampcoef)`; static default slot 1 ulp off at dts 18/4.5 s only |
| 2 composition with w44-constfold | `git merge-tree` HEAD + lane/w44-constfold `edec0ece58` (clean): CONSTFOLD=1 static == traced BYTE, CONSTFOLD=0 the 1-ulp class | CF01, 12+12 pass |
| 2c changed-value pristine WRF gate | `test_damping_pristine.py`: real PROD first acoustic stage, interior 4×4×44 crop (7 boundary-forcing leaves and the specified-W projection dropped: advance_w never reads them), original `calc_coef_w` + `advance_w` at dampcoef .2/.4 and zdamp 5000/7000; existing frozen bounds (RMS 2e-5, max 2e-4 rel) | PASS (FX02 FAST1 205 s; FX03 both tiers); both changes active in WRF and native, one ON trace. ON-default vs static (dt 18 s, 4×4×44 real crop): w max 2 ULP in 16/720 cells (max abs 2.44e-4 in acoustic-stage units, max rel 1.6e-7), ph max 2 ULP in 5 cells (9.5e-7), t_2ave identical — the dampmag slot class propagated through advance_w; changed values move w by up to 44 %/46 % (dampcoef .4 / zdamp 7000) |
| 2b GPU attribution | OPEN (GPU). Proposal: run the ON-default diffusion arm on the tree that contains constfold; it must then be BYTE vs OFF (the slot class vanishes). Only a non-BYTE result needs the dampcoef-excluded and diff6-only arms (each is a different program = extra cold compiles) | — |
| 3 rebase + OFF identity | Rebased (merge-tree clean). Candidate snapshot `5c4accf66d` (src tree `9cbdb4f9`, = later heads' src) | ID01: release_identity Swiss `b977712f` / PROD `321aa651` EXACT; o5-port schema-2 native-first probe (lane/o5-port `8e177c463e`, full-constant HLO, E224) base main `6e6e7a522c` vs candidate IDENTICAL — PROD nested 22/22 checks (root d01 `47c10960`, force-down `6be23bb3`, child d02 `5e5c3f00`; Pallas calls 161/29/151, 0 interpreted), Swiss 10/10 |
| addendum: whole-program value independence (production convention) | Same probe + one `--runtime-parameters` option (`ID01/tools/params_probe.py`, sha `92fa80e1`, 4-line diff), both families ON on PROD d01+d02: all 12 defaults vs all 12 changed | IDENTICAL 22/22 (root `3a5e0ab2`, force-down `6be23bb3`, child `929a9b9e`); non-vacuous: ON != OFF root/child, force-down unchanged; Pallas calls 161/29/151 = OFF (no new kernels), 0 interpreted |
| 4 docs/help | Index + docsite (`tools.html#runtime-parameters`, index link), help text, namelist comment, anti-damping band, oracle-tested MYNN values | DOC01 docsite build 20 pages, 2 search hits, links rewritten; test_docsite_links + test_cli pass |
| w44-release findings | ctopo CLI test updated to the authenticated behaviour (E76); `scripts/parallel_cases.py` `--runtime-parameters` CLI_ALLOWED arity 1 (manager 21:19Z) with an actual-child-argv test | FX02: test_ctopo_control 6, test_parallel_cases 47 pass |

## Default-value round-off rule

The manager's clarification makes OFF raw-HLO/jaxpr identity binding. An enabled
family at the static default may be byte-identical or a documented few-ULP class.
A non-byte result is kept only after localizing the difference to folding or
association of the newly traced constant, reporting max absolute/relative error
and affected real-column cell counts, passing a changed-value pristine WRF oracle
at its frozen tolerance, and proving one AOT key across values. A default-value
difference alone is not a shelving decision. GPU cost and E41 gates still apply.

The acoustic `dampcoef` slot has a specific pre-registered association class
(critic phase-A finding on 2026-10-24). The static path forms
`float32(float64(dts) * float64(dampcoef))`; the traced path forms
`float32(dts) * float32(dampcoef)`, matching the pristine WRF REAL multiplication
in `module_small_step_em.F:1284`. With `dampcoef=0.2`, these differ by one REAL32
ULP at `dts=18` (3.6 versus 3.6000001) and `dts=4.5` (0.9 versus 0.90000004);
they agree at 13.5, 6, 2 and 1.5 seconds. This describes the coefficient slot,
not a measured bound on the resulting forecast fields. Changed-value original-WRF
checks for both `dampcoef` and `zdamp`, real damping-layer max absolute/relative
error and changed-cell counts, and an actual GPU ON-default arm excluding only
`dampcoef` that is BYTE versus OFF remain required. A diffusion-only sixth-order
default GPU check also remains explicit because CPU identity does not establish
GPU division association. These gates are still open.

## Receipts and limits

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/w44-params/`.

- PA01: 6 pass/1 fail, detected a `None` versus empty-dict compatibility bug;
  corrected by 569c8a2226. PA02: 7 pass.
- PB01: 21 pass/2 fail. One test used the wrong key-helper signature. The other
  proposed a four-output-ULP compiled `w_damp` comparison that failed.
- PB02: compiled static and runtime `w_crit_cfl=1.0/0.5` results are byte-identical;
  literal evaluation is byte-identical to pristine Fortran at both values.
  Both compiled forms differ from Fortran by up to 0.0078125/0.01171875 in coupled
  tendency units; 18/1 cells exceeded the proposed output-ULP diagnostic. That
  diagnostic failure is retained. Compiled regression identity and literal WRF
  evidence are reported separately, with no relaxed numeric bound.
- PB03: native-at-trace OFF identity exact:
  Swiss `b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a`,
  PROD `321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b`.
- PB04: enabled default and changed-value Swiss production traces both
  `1c307a7359ead44fe4994ceb7adc39daf25e79348a9aa928940f3b92646d89b2`.
  These are CPU trace receipts, not GPU executions.
- PB05: 27 pass in 4.73 s. Pristine sixth-order checks retain the existing
  `tests/v025/b_diff/gate_manifest.json` limits and reject nonfinite values.
- PC01: 121 pass/1 fixture failure. A test attempted a one-domain checkpoint
  with a two-domain identity; E78 correctly refused. Fixture corrected to save
  both declared domains. MC01 regression: 33 pass after repair; no product-gate waiver.
- MC02: original WRF objects and extracted untouched predictor/Thomas routines
  agree byte for byte at SQFAC 3; changed SQFAC 6 passes the existing BP116
  64-epsilon profile bound on 12 daytime and 12 nighttime real columns.
  One test-only JAX `Literal` API issue was corrected; MC05 passes dependency tracing.
- MC03: old/new OFF whole-native CUDA-target raw HLO is byte-identical
  (1,323,317 bytes, `d399b0d9…`), including 12 native custom calls; E151 jaxpr exact.
- MC04: whole native MYNN compiled on CPU (Pallas interpreter), default 3 vs
  static: 25/25 leaves exact in each regime. SQFAC 6 changes one leaf with no
  additional JIT trace. This is not an actual-GPU byte claim.
- MC06: SQFAC 3/6 raw CUDA-target HLO exact (1,323,461 bytes, `a4ac100c…`).
- CS01: 14 pass, including changed `c_s` versus the unchanged original WRF
  `smag2d_km` routine on deformation derived from real Swiss winds and map factors.
- CS02: 5 pass in 3.16 s, including both actual periodic/specified callers:
  `c_s` reaches every consumer as a traced scalar and doubling it quadruples
  the Smagorinsky coefficient. Catalogue and CLI forwarding checks pass.
- OF01/OF02: full Swiss OFF raw CUDA-target HLO is byte-identical between the
  baseline and candidate `8aab900b63`: 22,468,110 bytes,
  SHA256 `3641e8bed499c14bfab5cde473e65b4283f2a6d26d93102446a2b7439d04c95e`,
  1,080 native calls, canonical identity `b977712f…`.
- OF03: candidate PROD OFF raw HLO matches the verified existing baseline:
  21,905,496 bytes,
  SHA256 `269260d3905000019a0ea6dfd74dab5e798930f4cb552ab5d4392d076ba02659`,
  1,088 native calls, canonical identity `321aa651…`. These are pre-topo proofs.
  The separately accepted topo fix changes the composed PROD identity to
  `6c88ea0d…`; final composition must compare against that baseline.
- GD01: immutable launcher input/source checks, child-process cleanup and
  bounded NVTX parser controls pass on CPU. No GPU execution occurred.
- GD02: 4 pass in 0.21 s. Fixed a diagnostic-only bug that replaced the reference
  field mapping after its first difference; multiple REAL32/REAL64 fields,
  signed-zero byte changes, NaN and Inf are now covered. Product source unchanged.
- MS01: ten conditional sensitivity arms pass finite checks, including byte-exact
  stable-function rebinding and entrainment packing controls. Selection above;
  no changed-value WRF or actual-GPU claim for the four proposed additions.
- PH01: 2 reference-adapter checks pass in 0.29 s. Extracted WRF `phim` at
  `CPHM_UNST=16, am_st=6.1` matches the original WRF object bytes on 24 real
  surface inputs; perturbations affect only their respective stability branches.
- AL01: 9 reference-adapter checks pass in 0.59 s. Extracted original
  `mym_length`/BouLac at ALP1 0.23 matches the original WRF object on 32 real
  44-level active-plume columns. ALP1 0.253 changes length and leaves qkw bytes
  unchanged. PH01/AL01 qualify the reference adapters; candidate comparisons
  are recorded separately in W301 below.
- EN01: 2 reference checks pass in 3.43 s. An unchanged full-source WRF rebuild
  reproduces all six selected DMP arrays from the original object; changing only
  entrainment 0.33 to 0.363 changes active plume fluxes. The existing real-column
  input emitter/caller is reused. Its separate downstream analytic-Kh tendency
  demonstration is excluded from this reference's scope.
- W301: 12 pass in 22.34 s on `fd4a66434f`. Stability coefficients pass the
  original BP116 64-epsilon QKE bound on 24 real columns; ALP1 passes the existing
  mixing-length bound on 32 real columns; entrainment passes the existing 5%
  DMP flux gate. All three compiled component default comparisons are BYTE.
  Values reuse their ON trace and AOT key; names remain bound. Existing SQFAC
  and cold-start consumers pass.
- W302: 2 pass in 3.46 s. Each of the four independent scalar inputs reaches
  native Pallas operands and outputs. First-call ALP1 is default BYTE and changes
  QKE at 0.253 with no additional value-dependent JIT trace.
- W303: complete native MYNN OFF raw HLO and canonical jaxpr remain BYTE versus
  the previous source, including 12 native calls: 1,323,317 bytes, `d399b0d9…`.
- W304 Swiss: full OFF raw HLO remains BYTE at 22,468,110 bytes, `3641e8be…`,
  canonical `b977712f…`, including first-call initialization in the production
  program. This proof uses the same immutable tool and source-location alias.
- W304 PROD: full OFF raw HLO remains BYTE at 21,905,496 bytes, `269260d3…`,
  canonical `321aa651…`, with 1,088 native calls. The proof is pre-topo; final
  composition still requires the accepted post-topo baseline.
- W305: all five MYNN operands together give 25/25 compiled output leaves BYTE
  at defaults in each real cohort; default max absolute/relative differences and
  changed-cell counts are zero. Joint +10% changes 14 daytime and 10 nighttime
  leaves, with exactly two traces (OFF and ON), and no value-dependent retrace.
- W306: combined ON CUDA-target HLO is BYTE across defaults and joint +10%:
  1,324,920 bytes, SHA256 `078442bef7541252e2c88b7fc094b3bfa5e23f0eb45621e952563737fad1668d`,
  canonical `10c70857…`, 12 native calls. This is CPU lowering, not actual GPU AOT execution.
- W307–W310: direct-consumer coverage totals 975 pass, six skips and one expected
  failure. The first batch incorrectly applied release flags to legacy fp64/dummy
  fixtures and failed five checks. The intended legacy setting cleared those;
  a later batch timeout was resolved by fresh per-file processes. All failures
  and partial results remain in the ledger; no source or tolerance was changed.

## GPU handoff

The previous `b1659f731b` launchers are withdrawn. An audit after the Thompson
lane's failed run found that our unrun root harness also omitted the history
observer, selecting a different fused program. No model source changed.

The replacement source/tool snapshot is `fa71118c9e` (model source tree
`6d399608e742d876f1722857baed049bebcc34bd`, implementation `fd4a66434f`).
`NIGHT_LAUNCHER.json` records eight domain/value preparation entries and two
family warm-measurement entries. All use one shared GPU cache; the two families'
OFF keys are identical. Prepare OFF/ON × d01/d02 before the corresponding warm
entry. Only manager-provided `GA_START_UTC` and `GA_END_UTC` are needed; there is
no evening-only date guard. Invoke directly as the sequencer child, with the
controller outside RAM wrappers; internal CPU stages acquire their own tickets.

Each cold GPU phase is capped at 900 s (whole entry 960 s). The cap uses the paid
Thompson OFF-root proxy: about 249 s for JAX compilation, including 201.789 s
backend time, followed by a 150.795 s export interval. These are proxy timings
from the larger, incorrect fused program; corrected full-grid GPU times remain
unmeasured. Warm measurement and profile GPU phases are capped at 300 s each,
CPU E41 at 600 s, and the whole warm entry at 1800 s. These are bounded ceilings,
not claimed GPU timings.

Host checks cover outer-PG-only TERM, nested timeout and escaped TERM-ignoring
children, including an actual RAM scope, with no surviving worker and a
reacquired private lock. Both final entries pass dry-run and missing/short/future
window checks; an unsupervised internal GPU helper refuses. SM01 proves the
actual production scheduler selects d01/d02, including the 2+1 and 1+2 child
history-alarm splits. SM04 verifies all 12 real-PROD preparation/runtime metadata
records and value-independent ON keys, including child force-down placement.

The replacement entries passed eight cold and four warm/profile child smokes.
Each captured the product `JAX_PLATFORMS=cuda` environment before substituting
CPU only at the device layer, then executed real Swiss diffusion/Smagorinsky or
four real 44-level MYNN columns. AOT export/reload was byte-exact; warm defaults
were byte-exact and changed values affected outputs on the same executable.
Measured CPU compile/export times were 0.246–0.278/0.029 s for diffusion and
6.585–6.923/0.349–0.364 s for MYNN. Subsequent preparation entries loaded the
already cached mini executable. Warm/profile entry walls were 5.34 s and 8.79 s
for the two families. These are launcher/mini-model timings, not GPU forecasts.

SM05 has 15 passing host controls; SM06 rejects a deleted actual ON artifact and
refuses to promote CPU records into GPU readiness; SM07 has 30 passing actual
entry-window refusals. Per-step `LAUNCH SMOKE PASS` lines and measured-cap receipts
were posted at 03:20:57Z, before the 10:30Z deadline. CPU compile caches and the
temporary check snapshot were removed; the frozen launcher snapshot remains a
live manager dependency.

No GPU arm has run or been self-started. Actual GPU default differences, one AOT key across values,
E41, fusions/transposes, device operations/time and memory remain to be measured.

### Successor GPU plan (2026-10-24 takeover; replaces the fa71118c9e entries once the FIX head is accepted)

- Tree: the critic-accepted head composed per the manager's order (budget first; with lane/w44-constfold present the
  dampcoef default slot is BYTE by construction, CF01) — never the pre-rebase fa711 source.
- Per family, d01+d02, one shared cache: OFF (release program), ON-default (cold compile), ON-changed (same executable,
  0 compiles, AOT key unchanged). Gates: ON-default vs OFF byte-identical carries/outputs (on a constfold tree also for
  dampcoef; otherwise exactly the registered slot class), changed != default, E41 on the ON programs, fusion/transpose
  counts vs OFF, warm root A/B n >= 10 interleaved in fresh cache-loading processes (E179/E191), VRAM per PID.
- Only if ON-default is not BYTE: the dampcoef-excluded and sixth-order-only attribution arms (each a different name set
  = a different program = extra cold compiles).
- Launcher: same builder as production, explicit step-alignment assert (o5-coast check_alignment.py; the CLI dry-run does
  not check it, LV01), every phase (model, E41 scan, checker) in the CPU rehearsal. A CUDA mini of a new program pays a
  full cold compile even on a cropped grid (FINDINGS E225), so its GPU time is the mini-grid compile, not <= 2 min.

## Integration

Branch-owned restart changes from `1ef11b7732` were composed exactly in
`328724e9f8`. Parameter digests authenticate values. The branch's static namelist
view must additionally replace only those controlled values with stable schema/shape/dtype
tuple tokens, because the ordinary full namelist digest hashes traced children too.
Names, shapes, dtypes, CTOPO2 and all other static fields remain bound.

The AOT static marker includes selected parameter names: the carry-key helper
excludes namelist structure, so a generic ON marker would collide for different
same-length scalar subsets. Fixed in `4a1993363f`, with a negative control.

Operator usage and restrictions: [runtime-parameters.md](../../docs/runtime-parameters.md).
