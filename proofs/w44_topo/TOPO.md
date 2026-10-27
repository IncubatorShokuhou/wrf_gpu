# Nested terrain shading audit

Status: **CPU operator, integration, OFF and exact-launch smoke gates pass;
GPU gates pending**. Native execution and WN3 forecast/twin gates remain open.
The manager's sequencer owns GPU launch. The revised 40-minute launcher is
admitted for the 2026-10-24 day/night queue, subject to the actual GA window.
No full READY claim is made.

WRF uses two independent switches: `slope_rad=1` enables the surface slope
correction; `topo_shading=1` supplies the terrain shadow mask. A shadow mask has
no surface-flux effect with `slope_rad=0`. There is no WRF `slope_rad=2` mode
(`Registry.EM_COMMON:2631–2632`).

## What pristine WRF actually passes

`module_radiation_driver.F:3506–3583` runs shading at radiation cadence. It copies
terrain into `ht_loc`, loads a two-row nested boundary into `ht_shad`, calls
`toposhad_init`, then calls `toposhad`. The search length is
`INT(shadlen/dx + 1)` grid steps, default `shadlen=25000 m`; this is a directional
grid-step limit, rather than a circular distance cutoff. With one whole-domain
patch there is one iteration. MPI patch iteration propagates shadow heights
through two-point halos and is a separate decomposition concern.

`toposhad_init` retains the outer two rows of HGT_SHAD. Where they exceed the
child's terrain it marks the point shaded and raises its local shading height.
`toposhad` returns both SHADOWMASK and HGT_SHAD, and clears the mask at night.
Its solar hour-angle formula includes the radiation midpoint and omits the
equation-of-time correction used by the separate `calc_coszen` routine.

The input field is **shadow height**, not a static extension of parent terrain.
The Registry declares `ht_shad` with `rdf=(bdy_interp:dt)`, and generated
`nest_forcedown_interp.inc:402–425` routes it through `bdy_interp` (default SINT).
However, `bdy_interp1` stores the **previous child** field in BXS and the
parent-to-child difference in BTXS. Radiation's `spec_bdyfield` call reads BXS,
without BTXS or a time interpolation. No consumer of the HT_SHAD boundary
tendencies was found in the pristine source audit. Consequently, injecting the
current parent SINT result directly into radiation is not established as
WRF-faithful behavior.

## Evidence

TP03 [M CPU], source `338e2d4d60`: the isolated JAX REAL4 operator matches verbatim
pristine `toposhad_init`/`toposhad` on WN3 0227 d03 (93×111, 1 km), at fixed UTC
hours 07:30, 08:00, 09:00, 17:00, 18:00 and 23:00. It uses the real d02 shadow
height interpolated onto d03 as an **explicitly injected operator input**.

- All six shadow masks match exactly.
- Maximum height error: 0.005555 m; registered bound 0.01 m + 2e-5 relative.
- Maximum SWNORM error against pristine `TOPO_RAD_ADJ`: 0.000367 W/m²;
  registered bound 0.002 W/m² + 2e-5 relative. Horizontal SWDOWN is a fixed
  600 W/m² caller operand, not a new full radiation-solver validation.
- Removing the injected input changes 41 cells at each of five daytime angles;
  the night mask stays zero. This demonstrates operator sensitivity, not the
  actual coupled parent's influence.

TP04 [M CPU], source `34bbfceb63`: verbatim, bounds-checked `bdy_interp1`, `SINTB` and
`spec_bdyfield`, using real WN3 d02/d03 shadow heights at 07:30 and 18:00.
Adding 1000 m to the parent height changes maximum boundary tendencies from
1.284 to 19.802 m/s, proving the parent input reaches the builder. The values
passed from BXS into radiation remain **exactly equal to the old child height**
in both arms (maximum difference 0 m).

Receipts and source hashes are under
`<USER_HOME>/wrf_gpu2_lanes/w44-topo/TP03/receipt.json` and `TP04/receipt.json`.
Reproducible drivers are in `proofs/w44_topo/`. The staged project input copy is
read-only; no ALISIOS data paths were read.

TP05: 105 existing catalog and namelist tests pass. TP06: dropping the parent
input from the compiled Python candidate produces genuine numerical assertion
failures at both 07:30 and 18:00 (41 mask cells each); 18 NaN/±Inf controls on
candidate/reference masks, heights and fluxes are rejected. TP07 proves the
experimental function AST is identical after moving it out of `src/`, and all
forecast runtime source remained byte-identical **before runtime wiring**. The only product AST change was
the explanatory text for an unsupported `slope_rad` value. Existing Swiss
`b977712f` and PROD `321aa651` numerical identities were carried forward for
that historical documentation-only candidate. The wired candidate's new OFF
checks are recorded below; TP07 is historical.

To reproduce on CPU, set `JAX_PLATFORMS=cpu`, `GPUWRF_JAX_CACHE=0` and
`PYTHONPATH=src`. Run `proofs/w44_topo/oracle.py --out <lane-artifact-directory>`
and `proofs/w44_topo/caller_probe.py --out <different-lane-artifact-directory>`
with Python, under the repository CPU affinity, QUIET and RAM-slot rules.
`--wrf-root` and `--case` override the source/input paths; `FC` selects the
Fortran compiler. Run `mutation_probe.py --fixture <oracle-out>/fixture.npz
--out <receipt.json>` in the same environment. Do not substitute the injected
operator result for the actual-caller result.

## Actual temporal behavior and clock correction

The manager's 18:01 decision selects actual pristine behavior and excludes
current-parent injection. TP10 follows initialization, SINTB, specified boundary
copy and shading over 145 nominal half-hour calls per nest (72 h). BXS equals
the child's preceding height at every call, independently of a +1000 m change
to the current parent. With equal solar geometry the released scan misses
one persistent d03 edge cell on 44 calls on days two and three; d02 has no
equal-geometry misses in this case.

TP11 isolates a second bug: WRF `toposhad` omits equation-of-time, while
`calc_coszen` supplies equation-of-time to both RRTMG and `TOPO_RAD_ADJ`.
The port reused that RRTMG geometry for the shadow search. Its RRTMG/slope
clocks already match the pristine oracle (max cosine error 1.78814e-7,
hour-angle error 4.76837e-7 rad, declination error 1.49012e-8 rad). The fix
separates only the shadow clock. All 42 tested instantaneous masks then match.
The standardized 600 W/m², diffuse-fraction .3 transfer experiment measures a
response coefficient, **not actual forecast irradiance or forecast error**;
its large low-sun flux values must not be presented as real-world impact.

TP10's first replacement failed the unchanged height tolerance near sunset.
Preserving WRF REAL assignment rounding fixes that failure: TP12 (JAX CPU)
and TP13 (Pallas CPU interpretation) each pass all 290 temporal masks exactly
and have zero height failures. Maximum errors are 0.005188 m (d02) and
0.005127 m (d03). TP14 lowers the actual CUDA/sm120 branch for root and nest
without a GPU: 128 lanes, four warps, no fp64 in either Triton IR. This is
compiler preflight, not GPU execution evidence.

## Implementation and cost

The new Pallas kernel keeps each cell's complete ray loop inside one kernel.
An optional `shadow_height_m` leaf in the existing held radiation diagnostics
retains HGT_SHAD. Cold static nests start at allocator zero, as WRF does before
the parent's first solve. Radiation refresh passes the child's own previous
height; there is no new timestep force-down or new State/OperationalCarry field.
The SW solver's topology flattening omits the height, which belongs to the
coupler rather than the column solver. HGT_SHAD stays fp32 after diagnostics
conversion. OFF keeps the optional leaf absent.

The three WN3 planes contain 49,962 REAL values, or 199,848 held bytes [I from
shapes]. Actual GPU timing, temporary memory, launches/copies and compile cost
are pending. No new host/device transfer appears in the device loop in the
source audit. New restart markers require a finite fp32 plane of the correct
shape and refuse old shading-enabled checkpoints that lack it. Goddard and
FLG/GFDL keep this unsupported optional field absent.

TP15/16 pass 27 focused lifecycle/topography/serialization tests, including a
compiled refresh/hold/refresh branch. The 134-file shared-consumer roster is
covered by TP17, its corrected/unrun-file delta TP21 (175 tests), the three
focused files and TP27 (99 AOT cheap-key tests, 467.46 s). TP17's 180-second
AOT timeout and suspend-truncated batch are incomplete results, not passes.

Rebase onto `c653483682` incorporates raddiag's generic handling of every
unproduced optional field. Our Goddard initializer and both scheme-validator
hunks were dropped; all three production files match that upstream pin.
`shadow_height_m` follows `rad_diag`, and the shadow validators follow the
scheme-specific validators. TP25/26 pass all 122 unique integration tests;
one duplicate-field test setup was corrected without changing product code.
The ray kernel and shadow clock are byte-identical to the sealed native probe.
TP32 deletes retention at the real driver helper: resetting the incoming height
to zero makes the compiled refresh/hold/refresh test fail numerically (one
genuine assertion, zero setup errors). TP31's static transfer scan has zero
findings across the three new shadow helper modules.

## Program identities and intentional default correction

The manager accepted the default-program change on 2026-10-23: PROD
(`namelist.input:66–67`) and WN3 0227 (`:70–71`) already select
`topo_shading=1, slope_rad=1`. They receive this WRF-faithful bugfix.
The Swiss case disables shading and must retain its exact identity.

TP23 [M CPU canonical native-at-trace d01 program]:

- Swiss: `b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a`, exact.
- Corrected PROD: `6c88ea0d73eca94e590270e250f8f0122416388ef9a62e4db01d2a093a565be3`.
  Its previous shading-enabled identity was `321aa651…`; this change is intentional.
- Corrected WN3 0227 d01:
  `decf559fdc8d792df5345b6b18e72eeff2b2f75006445feaf587d365883af395`
  (TP29, 14,741,195 canonical characters). Native d01 and fused-child HLO/cache
  identities will also be recorded from the GPU receipt; a root trace is not
  evidence for the complete nested executable.

TP24 [M CPU]: with **only** `topo_shading=0` and `slope_rad=1` retained, old
`748191e1eb` and wired `a2e05617dd` have the same canonical program:
`96ee8df37ac9ec220aa54a26d60db9745a95e9e914df780b1c8ff1b1c069cc7e`,
14,705,071 canonical characters in each. Original inputs were unchanged.
This proves the required shading-OFF invariant, rather than treating an ON
program as OFF. TP29 confirms final rebased Swiss/PROD exactly and records WN3.

## Sequenced forecast effect arm

The original native operator launcher at
`<USER_HOME>/wrf_gpu2_lanes/w44-topo/night_launch.sh` is immutable, SHA256
`d949c6025ed8a1a71f0b331b894419c92c7399d076e5976ad780ad2725bb3c91`.
Its intended scope was 24 pristine-oracle cases, bounded operator timing and
real-cache E41 in a three-minute cap. The manager attempted it at
2026-10-23T22:35Z: official lock acquisition timed out after 1.161 s before
JAX/model code; the subsequent empty-cache E41 call returned an error. There
were zero measured GPU cases and no executable. This is an infrastructure
refusal, not a scientific failure. Both child process groups were cleaned.
The original files and failure receipts remain unchanged. TP34 records it.

The combined launcher includes that oracle, then old/new WN3 runs from their
unchanged **00Z** inputs through +11 h, reporting the **08–11Z** daylight slice.
Its immutable path is
`<USER_HOME>/wrf_gpu2_lanes/w44-topo/night_forecast_v3.sh`, SHA256
`d73606ad0b206c13cd47726cd375df7543da89242330ded8e15c3f737e2f0116`,
cap **40 minutes including the native oracle**. Tools are committed at
`2c2d1a1a8a`; the model snapshot remains `cf1236c9f2`. Only `GA_START_UTC` and
`GA_END_UTC` are required. It refuses a window with less than 40 minutes left,
takes each official GPU lock with `--timeout 1`, and kills each owned child
process group on timeout. It has no date or evening-only guard. v3 supersedes
the original 40-minute `d012d000…` entry and the standalone native entry;
those immutable files remain untouched. It skips E41 when the native operator
never executed, and writes durable receipts after every completed phase.
`NIGHT_LAUNCHER.json` records the final seal and exact-entry smoke receipt.
The first three hours are night and cannot measure daytime shading impact.
The old arm consumes the sealed v0.4.3 bundle through its official COW layer;
any new old-arm compile refuses reuse until a separate complete E41 check.
The old bundle's existing optimized-HLO gate is recorded as INCOMPLETE,
never renamed PASS. The new candidate keeps full AOT metadata and must pass
the complete real-executable E41 gate including declared nested programs.

The effect reader rejects nonfinite old and new data and incomplete/misaligned
history. It reports SWDOWN/T2 from real history and GSW recovered from held
`swnorm - swup_topographic` at output boundaries. The GSW capture adds host
transfers/IO only at outputs; it changes no compiled production function or
carry leaf. Its timings are disturbed by that observer and are not S1 claims.
Old/new comparisons measure forecast effect, not CPU-WRF fidelity. Receipts
retain source/input/tool hashes, asserted GPU backend, finite coverage, model
program inventory, E41 and per-domain/max/RMSE/mean differences. Successful
arms remove their disposable caches and raw outputs immediately after scoring.
TP30 [M CPU] validates the actual callback wrapper, its timestamp and original
writer delegation, six observer nonfinite controls, reader baseline/one-ULP/
old-and-new nonfinite controls, 36-frame coverage with the expected 12-frame
daylight aggregate, a missing-capture refusal, four UTC-window controls and
timeout cleanup of a child/grandchild sharing a private flock. These fabricated
IO/process fixtures are launcher tests, not scientific forecast evidence.

TP35/36 apply the manager's stronger per-step admission rule. The exact sealed
entry runs all three actual child payloads, including the official old-bundle
COW consumer and the real S0 argument parser/CLI pre-exec. A single argv/env
builder serves GPU and smoke modes. The last JAX-dark device boundary records
the product CUDA environment, then substitutes only `JAX_PLATFORMS=cpu` for
the explicit `--launch-smoke` path. The native probe interprets the same Pallas
kernel; forecast smoke selects a 4×4×4 RRTMG caller before the full forecast.
It makes no fake GPU device or GPU-PASS receipt.

The first smoke correctly refused a mini State lacking MP_RE radii. The fixture
now uses `State.ensure_conditional_leaves` with MP_RE=1 and RRTMG4/4; product
flags and model source are unchanged. Final exact-entry smoke TP36 [M CPU]:

- Native 24 cases: 4.042 s, all masks and registered height checks pass.
- Old S0/RRTMG child: 14.989 s; 16 finite diagnostic leaves; radiation first call 9.070 s.
- New S0/RRTMG child: 11.244 s; 17 finite diagnostic leaves; radiation first call 8.980 s.
- Whole entry: 30.674 s, all three pre-import child environments explicitly CUDA,
  actual devices CPU, original bundle BUNDLE_OK, zero surviving descendants.
  Disposable CPU caches/overlays were removed. Scientific GPU gate stays PENDING.

TP37 passes four final-entry GA controls (including a valid pre-19Z window),
and rejects product `gpu`/`rocm`/`cpu` aliases before JAX import. The required
`LAUNCH SMOKE PASS w44-topo … CAUSE/FIX: … CAPS: …` line was posted before the
10:30Z day-slot deadline. Receipt:
`<USER_HOME>/wrf_gpu2_lanes/w44-topo/LAUNCH_SMOKE_RECEIPT_v3.json`.

Caps use disclosed phase evidence: [M GPU proxy] WN3 3-nest cold 3 h was
943.214 s (207.667 s to first segment, first segment 718.604 s including
390.010 s compile; warm h2 5.725 s/fc-h); the v0.4.3 warm WN3 3 h receipt is
77.170 s. [I] Extending the cold proxy by eight warm hours gives 989.014 s,
with a 1,400 s candidate cap. Phase caps are native 120 s + native E41 30 s,
old warm 600 s, new 1,400 s + E41 120 s, effect reader 40 s, all within the
2,400 s outer limit. Exact candidate GPU and E41 phase costs remain [U];
the smoke's CPU times do not replace those measurements.

## Critic phase-A fixes (cr-w44-topo FIX 2026-10-24T15:39Z)

F1 [M CPU]: both production nested init sites (`integration/nested_pipeline.py`
`_load_domains`, `nesting/initial_feedback.py` `initialize_two_way_domains`)
replace `radiation_diagnostics` with the lead-0 Noah-MP init radiation call.
That call is not a WRF toposhad history call, so the seeded allocator-zero
HGT_SHAD is now re-applied after the replace (`coupling.topographic_shadow.
preserve_initial_height`, reuses the seeded plane, no allocation; OFF returns the
init diagnostics object unchanged). Critic arm CR02 showed the defect on an
18:30Z d02 start (one edge cell shaded all next day, +2.07 m held); 00Z starts,
including every WN3 gate case, were unaffected. No compiled-program change.

F2 [M CPU]: in-repo frozen pristine fixture
`tests/v025/w44_topo/fixtures/shadow_d03_regression.npz` (real WN3 0227 d03,
verbatim pristine REAL routines, 30-minute cold-zero trajectory: sunrise,
sunset, first night call with a held edge cell, next sunrise). The pytest checks
the toposhad clock, CPU reference, the nested coupler, the CUDA dispatch wiring
and the Pallas kernel (interpret) against it; masks exact, heights at the
original bound; nonfinite candidates rejected. `test_shadow_init_seam.py`
executes the real init statements of both sites (AST) with only the radiation
call stubbed.

F3: the decomposition halo test runs the shipped operator
(`topographic_shadow_from_geometry`); the old mask function stays only as a
labelled legacy comparator for historical proof drivers.

F4 (addendum A1): the lifecycle fixture now seeds the release-flag leaves
through the production cold-init seam (`State.ensure_conditional_leaves`), so
the EOT-guard test runs under release defaults. [M CPU] tests/v025/w44_topo +
the halo test: 40/40 PASS with `GPUWRF_FAST_DEFAULTS` unset and with `=1`.

Deletion mutants (TP48, `proofs/w44_topo/fix_mutants.py`, flags unset): 8/8
killed by genuine assertions, 0 setup errors: M1 shadow uses RRTMG EOT geometry,
M2b coupler drops the nest's held height, M3 kernel night height `h`, M4 refresh
passes zeros, F1a/F1b re-seed deleted at either init site, M5 CPU reference night
height from raw terrain, M6 clock adds the radiation half-step twice. Receipt
`<USER_HOME>/wrf_gpu2_lanes/w44-topo/TP48_FIX_MUTANTS/receipt.json`; the same
sweep under release defaults is TP49.

## v6 launcher (2026-10-24/25, supersedes v3/v4/v5)

v3/v4 ran old and new with `--hours 11`. WN3's root dt is 54 s, so 11 h is
733.33 root steps and `nested_pipeline` refuses the run after init (E187). The
v3 CPU launch smoke had replaced the model hook and never reached that check.
v3/v4 are revoked.

The v5 real-model CPU smoke (TP50) then found a second latent bug: the driver
read `e41_old.json` as a file, but `wave_cubin_gate.py` writes a receipt
directory (`<out>/wave_gate.json`). Before that point TP50 passed the native
oracle (24/24 exact) and the real v0.4.3 old init on CPU through the 12 h
alignment check (703 s, 517 finite leaves). v6 reads the receipt directory.

v6 (tools commit `4e0d1b27ec`, request `FORECAST_REQUEST_v6.json`):
- production 12 h (800 root steps; report 08–12Z), mini 3 h (200 steps). The
  driver preflight refuses any profile whose hours are not a multiple of 3 and
  whole root steps, before any GPU work; o5-coast `check_alignment.py` agrees
  (12/3 accepted, 11/1 refused).
- CPU launch smoke runs the real model hook and CLI on CPU through case load,
  cold init and the run-length check. It stops at the first post-init call
  (`maybe_prewarm_defused_nest`) and records the real carries, including each
  nest's HGT_SHAD (must be float32 zero: F1 on the real case). Only device
  admission differs from the GPU child; RAM comes from the caller's
  `with_ram_slot` cgroup.
- CUDA mini: same builder and case, phases native + E41 and old v0.4.3 warm
  3 h + E41. The new phase is not miniaturisable: its cost is the cold 3-nest
  compile, the same as in production.
- Effect reader adds history SWNORM; GSW is labelled reconstructed LSM net SW.

## v7 launcher (2026-10-25, supersedes v6 after launcher QA)

Launcher QA (cr-w44-launchers, 01:06Z) found that v6 would have stopped in the
GPU window, and both CPU smokes missed it:
- B1: `wave_cubin_gate.py` returns rc 2 for INCOMPLETE. That verdict is normal
  whenever compact AOT step programs are scanned (E201/E230).
  - The old phase scanned the v0.4.3 bundle cache. Its seven AOT entries give
    INCOMPLETE, so the driver aborted after the old phase.
  - The new phase would also have stopped at its E41, before the checker.
  - The CPU smokes hid this because they scanned a JIT-only fixture instead
    of the real cache.
- B2: a bundle-size scan takes about 3 min, more than the mini's E41 cap.
- B3: the CUDA children had no memory scope.

v7 (tools commit `d02a77532c`, request `FORECAST_REQUEST_v7.json`):
- The E41 record is `<out>/wave_gate.json`, never the tool rc.
  - PASS, or INCOMPLETE with STACK PASS and no callee/kInput flag, is
    acceptable.
  - FAIL, or MISSING (tool error or timeout), is recorded and the measured
    phases continue.
  - Final rc: 0 when every verdict is acceptable, 2 for an E41 FAIL, 6 for a
    MISSING verdict (that cache is kept for a CPU scan after the window).
- The sealed v0.4.3 donor is not rescanned. Its record is the dependency-hashed
  `bundle/e41/wave_gate.json` (INCOMPLETE, STACK PASS, max 32 B). The bundle
  consumer's receipt must also show that the warm GPU run compiled no new
  program (`compiled_new == 0`); otherwise the record is not acceptable and the
  layer is kept for a scan.
- The new-cache E41 runs after the checker, so a slow scan cannot cost the
  effect numbers. Its cap is 400 s, bounded by the driver cap.
- Every model child (native/old/new) runs in its own `systemd-run --user
  --scope`, with MemoryMax 16G (12G in the smoke) and MemorySwapMax=0 (E233).
  - The driver probes the scope before any GPU work.
  - `gpu_dispatch.py` asserts the effective limits before taking the lock and
    records them.
- v8 (tools commit `402664ffc1`, request `FORECAST_REQUEST_v8.json`; v7 never
  ran) widens the old phase's outer cap to the device cap + 30 s. The bundle
  consumer's view and P checks run after the device phase inside that cap
  (11 s measured in TP52); v7 left only 5 s.
- The production CPU smoke runs the real gate on the real AOT-bearing v0.4.3
  bundle cache (read-only) for the new role, and requires INCOMPLETE with
  STACK PASS and rc 2. The native role keeps the retained CUDA JIT fixture,
  because the native GPU cache is JIT-only too.

## v9 re-pin and identity (2026-10-25, manager 08:10Z/08:15Z)

v9 (tools `477a5cd6f7`, `snapshot_v9`) is the v8 launcher logic re-pinned onto
origin/main `700a382f8b`.

- The candidate src tree `7f80f755` is main's src plus the reviewed topo diff
  only. The same 13 files carry equal per-file patch-ids
  (`REPIN_DIFF_PROOF.json`) [M git].

Schema-2 native-first probes on WN3 0227, nested (d01 step, d01→d02
force-down, d02 child step), with FAST_DEFAULTS=1 [M CPU]:
- OFF: with only `topo_shading=0`, main and candidate are IDENTICAL.
  - Full-constant HLO `e98a0b25`/`ce38cc34`/`eead2214`.
  - Triton calls 154/7/140, kernel bodies with and without locations equal.
- ON: candidate vs main is DIFFERENT, as intended. The change is exactly one
  new Pallas kernel, `wrf_topographic_shadow`, in d01 and d02 (155/141 calls).
  The force-down HLO changes because the held HGT_SHAD rides in the carry.
- Attribution: sealed v0.4.3 vs main, both with old topo ON, is DIFFERENT.
  - The force-down is identical, and all kernel bodies (without locations) and
    counts are equal.
  - The d01 and d02 step full-constant HLO differ.
  - So an effect measured against the v0.4.3 donor would mix in non-topo main
    changes. The v8/v9 effect design (old = sealed v0.4.3 warm) is NOT
    topo-attributable.
  - The follow-up must run the old phase on main `700a382f8b`, which has the
    old topo, with a cold compile and its own cache.

CPU smokes for v9 (own 12G scopes) [M]:
- TP56 mini: CPU_ALL_PHASES_PASS.
- TP57 prod: CPU_ALL_PHASES_PASS; the real AOT E41 gave INCOMPLETE, accepted,
  in 188 s.

Tests TP55:
- FAST_DEFAULTS=0: 104/104.
- FAST_DEFAULTS=1: the only failures are the 13 `test_restart_store` reds that
  plain main also has (deliberate guards, w44-release P1).

## Remaining gates

Native GPU oracle, nested finite/E41 and daylight forecast-effect numbers remain
open. The manager sequences the combined launcher in the granted GA window;
no GPU was started by this worker. The release candidate **must rerun RW3: WN3 0227 three nests,
72 h against original CPU-WRF, D6/A3**. There is no scientific carry-forward
for this non-bit-identical correction. Final release RW1/RW2/RW4 also apply.
