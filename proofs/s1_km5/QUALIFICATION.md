# SMS-3DTKE qualification (s1-km5)

The full operational km_opt=5 chain is implemented and CPU-oracle-qualified.
The manager accepts the coupled F2 disclosure in F2_DISPOSITION.json: terminal
guards match the km_opt=2 control, and post-RK QC visits15 versus14 reflect a
different closure trajectory in the same round-off class. The minima are
-1.0379390923132004e-11 versus-2.830937307490844e-12kg/kg. No full F2 waiver,
masking clamp or tolerance change is claimed. Current-main shared verification
passes362 checks; official release identity is unchanged. GPU binary, coupled
forecast accuracy and GPU performance remain unqualified. VALIDATION.json records receipts and source-specific reuse.

## Implemented operational scope

Selection is diff_opt=2, km_opt=5, bl_pbl_physics=0, single domain, surface
layer0 or5, and land0 or4. Surface layer0 requires land0 and retains held/
prescribed surface fluxes. Both pipelines bind the selection; CLI honour checks
reject unsupported pairings before running. MM5 surface layers1/91, RUC,
live nests, DFI and unsupported extra microphysics scalar families remain
explicitly refused. These callers need separate qualification; they are not
silently mapped to MYNN. WRF's CU11 prohibition and nondefault TKE/flux options
are honoured. The catalog and support matrix mark CPU-oracle qualification
and pending GPU/coupled-forecast qualification honestly.

The implementation follows pristine module_first_rk_step_part2.F:

1. Geometry, deformation and moist stability feed nonlocal_flux,
   free_atmos_length, meso_length_scale and SMS tke_km.
2. Nonlocal buoyancy enters tke_rhs, and its dissipation length feeds the TKE
   source. Implicit vertical momentum/heat/TKE/moisture matrices use literal
   Thomas solves, WRF's doubled TKE factors, surface/lid rows and the original
   exclusion of precipitation-number vertical mixing. Horizontal terms remain.
3. One frozen RK1 bundle feeds dry and scalar sources through all three RK
   stages. State.tke is the separate prognostic tke_2, distinct from MYNN qke.
   The existing PD scalar transport, mass coupling, bounds and flow-dependent
   boundaries are retained.
4. Surface layer/Noah-MP export BR, RMOL, WSPD and USTM. GAMU/GAMV retain WRF's
   inactive-column values. Gamma, PBLH and HFX survive the RK origin reset.
   Six optional REAL surface fields are cold-seeded before tracing and held
   thereafter. Runtime checkpoint, full-carry pickle and NetCDF restart retain
   the six fields and TKE byte-exact, reject each missing active field (E78),
   and keep legacy inactive formats optional.

USTM is distinct from UST: MYNN water uses the relaxed raw-wind value
(module_sf_mynn.F:955), while land assigns USTM=UST (:961). Nonlocal flux uses
UST; dissipation/implicit diffusion use USTM. The full fixture deliberately
uses USTM=0.7*UST to expose an alias. The surface-driver MYJSFC assignment
USTM=UST is not the MYNN rule.

## Independent numerical evidence

Private copies of unmodified pristine WRF component libraries supply the
references. The shared instrumented solve_em executable was not used as
truth. Four tracked manifests bind source/library/fixture hashes:
fixture.yaml, upstream_fixture.yaml, full_fixture.yaml and stage_fixture.yaml.
Fixtures retain Swiss12x10x44 structure at400/600/1200m, specified/periodic
boundaries, mixed/convective/zero-TKE stable regimes, isfflx0/1/2, active
graupel and separately prescribed real-structured Ni/Nr. The standalone
coefficient fixture also covers80/3000m and high TKE.

- Coefficients/dissipation:36 eager/JIT checks pass,14.70s [M CPU,K5-03].
  The qualified functions remain byte-exact; only common_gradient was added
  (K5-29 source proof). Three genuine numerical mutants are killed.
- Full operator:13 tendencies and4 surface updates pass three regimes
  [M CPU,K5-22 and final-source K5-64]. The registered tendency bound is
  rtol32eps32 plus atol32eps32*maxabs(entry)*maxabs(mass)/dt, accounting for
  cancellation in mass*(implicit_solution-entry)/dt. Surface and upstream
  fields use unchanged strict32eps32 relative plus32eps32*maxabs(reference).
  No bound was widened. Full fixture regeneration is byte-exact (K5-23).
- Compiled full operator passes on the composed source (K5-64). Deleting the
  actual implicit solve yields a numerical AssertionError, with zero setup
  errors [M CPU,K5-58], in addition to the three coefficient mutants.
- Three own steps/nine RK stages evolve TKE/gamma under frozen real Swiss
  wind/mass/moisture/surface forcing and zero eta transport. Final-source
  K5-64 passes all stages, maximum TKE error5.6028366e-6 at the unchanged bound.
  This isolates SMS/scalar wiring; it is not full forecast fidelity.
- Nonlocal stage localization compares23 observed stages byte-exact with
  pristine at both problem columns (K5-17). The original CPU rounding helper
  was ordinary arithmetic; the ring-relax REAL emulator fixes that mismatch.
  The final explicit-REAL mesoscale gradient fixes one periodic DLK failure;
  all seven upstream outputs pass on that case at the unchanged bound
  [M CPU,K5-60]. ELMIN at the failing point was already byte-exact (K5-59).

The initial v0.3.4 NaN came from zero absolute pressure in an idealized caller:
OFF and old-SMS arms both had U/V NaNs while all13 turbulence intermediates
were finite; changing only p/p_total to90000Pa made both finite (K5-01).
A second legacy fixture used constant C3F, causing192 zero-density cells.
Correct eta pressure coefficients give density1.5373..1.7021kg/m3, agreeing
with independent WRF arithmetic within2.0573e-7; both legacy callers pass
(K5-52/53). These are fixture corrections, not model clamps.

## Coupled result and blocker

Final source d5eb28995, main40799926d: K5-63 completes three realSwiss42x42x44
steps atdt18 in472.2s after loading, including CPU compilation. Resolved work
is MP8/CU0/SF5/Noah-MP4/RRTMG4/PBL0/diff2/km5. Carry and metrics are REAL32;
global x64 is enabled for CPU REAL-rounding emulation. All189 leaves are
finite. TKE maxima are0.7330819/1.1374544/1.4220737m2/s2 and PBLH maxima
729.1948/711.3933/695.1625m.

Steps1/2 have no F2 events. Step3 has15 post-RK QC negatives, minimum
-1.0379391e-11kg/kg, and three surviving outer-ring negatives after MP,
minimum-7.8876955e-14kg/kg. There are no over-cap values. Guards are disabled;
every actual added/removed/undefined guard-water counter is zero. The legacy
three repaired labels mean would-repair candidates. The original zero-candidate
assertion FAILS; the later manager decision accepts these specific round-off
observations with disclosure for CPU lane graduation.
No epsilon, floor or mask change is introduced. Original WRF excludes the
specified ring from microphysics, matching the port. Initial QCLOUD and all
QCLOUD boundary values/tendencies are zero (K5-38/39).

K5-55 before the final gradient correction has the same QC population/minima.
That comparison is regression evidence only, not WRF truth or an F2 waiver.
The earlier timed-out controls have no numerical readout. F2_DISPOSITION.json records the later manager decision; no additional source
replay is required for this CPU gate. General/native release F2 qualification
remains separate. See COUPLED_BLOCKER.md for the complete disclosure.

The successful original-release control K5-75 uses the same inputs, dt18,
three steps, REAL32 precision, census and disabled guards, with the original
namelist (diff1/km4/MYNN5). It completes in238.6s after loading, including CPU
compilation, with0 nonfinite values across182 leaves. It has4 post-RK QC
negatives (minimum-6.2862831e-18kg/kg), and one surviving outer-ring negative
(minimum-3.7423635e-37kg/kg). Actual guard-water changes are zero. The input
hashes and original namelist versus the exact four SMS edits are sealed.

Main192b35d4f allows inherited guard events only when field, count and location
class match a release-path control. Counts3 versus1 do not match. Thus this
MYNN control does not justify the candidate population. The closer TKE
control below resolves the terminal guard classification separately. MATCHED_CONTROL.json retains both populations and the exact verdict.
This port-vs-port comparison is regression/attribution evidence, not WRF truth.
Neither CPU timing is a GPU speed result or a matched-work performance comparison.

The closer K5-78 control changes only km_opt5 to2, retaining diff2/PBL0/SF5
and every other resolved option, input, timestep and diagnostic setting.
Maina8da6cdeb independently GPU-qualifies km2 against original CPU-WRF24h;
this local control comparison is still attribution evidence, not SMS WRF truth.
It completes three steps in233.8s after loading, including CPU compilation,
with0 nonfinite values across183 leaves. It reproduces all3 terminal QC
candidates: same field/count/ring0 class and minimum index(k11,j41,i4).
Actual guard-water changes are zero. The standing terminal-guard policy PASSES.

The full raw phase/class comparison has one remaining difference: post-RK QC
counts14 forkm2 versus15 forSMS (interior11 versus12); minima are
-2.8309373e-12 versus-1.0379391e-11kg/kg at the same(k13,j38,i6).
The terminal minimum is-2.6254012e-14 forkm2 versus-7.8876955e-14 forSMS.
PAIRED_KM2_CONTROL.json records both classes without a full F2 waiver.
The manager accepts this specific post-RK count and amplitude difference as
a different closure trajectory in the same F2 round-off class (E213/D27),
with the measured minima disclosed. This closes the CPU lane gate; it is not
a general upstream amplitude proof or a full F2 waiver.

## Integration, performance and memory

After rebasing onto main0bac9f30d (urban and history integration), all six
SMS numerical modules remain byte-exact to the scientific candidate936f9d766
(RESUME_COMPOSITION.json). The final composed run K5-82 passes362 checks in
464.21s [M CPU], with0 failures/errors/skips: all8 required shared-consumer
files including the full operational physics smoke, NetCDF process locking,
all restart readers, changed history output coverage,5 SMS production seams
and tracked asset checks. All three restart paths retain each active leaf
byte-exact and reject missing active fields. Earlier source-specific numerical
receipts remain bound to the unchanged SMS modules; no numerical validation
was repeated for that byte-identical code. Clean14-file fixture/archive proof
K5-36 is retained, and final assets pass again in K5-82.

The official proofs/s1_port/release_identity.py on rebaseda1ecf3870 returns the full
frozen Swiss hash2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a,
13405317characters,47.7s [M CPUtrace,K5-83]. The old CPU-facade d339647f probe
is not the release identity. No GPU execution is involved.

The changed native artifact passes deviceless CUDA/sm120 TTIR lowering for
all16 SMS calls, zero fp64 [M CPUtrace,K5-66]. The portability facade must be
held native during this trace; the old probe failed on CPU reduce_precision
(K5-65). No product arithmetic was changed to fix that probe.

Column recurrences and searches stay inside kernels:4 upstream calls plus12
Thomas solves, with ordinary JAX matrix/stencil work around them. The final
meso correction replaces arithmetic with explicit REAL rounding in the same
kernel; it adds no launch or persistent allocation. Six surface carries add
24bytes/horizontal point; separate TKE adds4bytes/3-D point relative to a
closure without LES TKE [I]. No timestep host/device transfer was added.
CPU472.2s is not a GPU speed claim. GPU launches, traffic, peakVRAM, SASS/stack,
readiness and coupled forecast D6 remain unmeasured. The default program is
unchanged, but source cache keys may still require cold readiness work.

All Python arms use JAX_PLATFORMS=cpu, GPUWRF_JAX_CACHE=0, nice19, the granted
cores, a same-line QUIET guard, timeout and an immutable detached snapshot.
Only one pytest arm uses the exact lane basetemp at a time. Private reference
builds, disposable caches and the detached snapshot have been removed after
their last dependent arm. No GPU or ALISIOS truth-value access was used.
