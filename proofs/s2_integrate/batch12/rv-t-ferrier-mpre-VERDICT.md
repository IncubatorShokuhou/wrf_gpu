# MERGE — lane/t-ferrier-mpre

Reviewed commit `653e4b68de31e1028c2425bc18b2e99f021686f6` against unchanged parent source `d85aad784`.
Independent CPU-only review on an immutable detached snapshot, cores 12,13,28,29,
nice 19, one JAX process, RAM slots capped at 6 GiB and zero swap, QUIET guards,
300-second arm timeouts, JAX cache disabled. No production changes by this critic.

Pristine WRF 4.7.1 `module_physics_init.F:985–1027` requires eligible LW **and** SW
before activating Thompson's radius flags. RRTMG LW (4) with Dudhia SW (1)
leaves all three flags zero. Background radii are seeded only for active flags
(:1267–1300); Thompson computes/updates them only for active flags
(`module_mp_thompson.F:1466–1477`). Radiation receives the held state before
the current microphysics update. For this mixed pair no MP radius time level
is consumed: RRTMG LW uses its flag-zero optics path and Dudhia has no radius
arguments (`module_radiation_driver.F:2403–2436`). This corrects the contract's
shorthand about passing held radii: this configuration must keep them inactive.

The two-line production fix uses the existing `mp_re_active(namelist)` predicate
and explicit `mp_physics` at the SSiB column-builder boundary, matching the
standard operational refresh. No new seeding, recomputation, solver mathematics,
carry, restart fields, assets, or other LSM hooks. Eight physics/helper files are
byte-identical; all other functions in the changed hook are AST-identical.

Independent results [M]:

- RMP01: 3 passed, zero failures/errors; actual Dudhia + RRTMG LW runtime
  refresh twice for use_mp_re=0/1, finite outputs and radii remaining None;
  explicit MP identity and resolved selection forwarding checked. Constructed
  6×2×3 caller test, not a forecast-fidelity oracle.
- RMP02b: actual c40 full history segment lowers to CUDA, 72 steps/radiation
  cadence 12, 128 native Triton calls; 50.973 s CPU process wall,
  2,877,192 KiB peak RSS. Input, target, harness and full
  269,549,752-byte StableHLO SHA match producer evidence exactly. Physical
  backend stayed CPU; compilation/E41 not attempted.
- RMP03: restoring original hook kills exactly one test with the genuine held
  radius refusal; zero setup errors. Mutant harness rc0 attests the expected kill.
- RMP04/RMP05: official default Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` and
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` exactly preserved.
- RMP06: 5 pristine WRF file hashes/excerpts, 18 receipt hashes and 18 decoded
  hashes verified. Producer FAST1 15 failed nodes equal the unchanged-parent
  inventory; FAST0's 12 failures are its subset. These remain disclosed baseline
  failures, not passes or a general SSiB restart qualification.

The predicate and MP identity are Python-static [I/source]: no new kernel,
transfer, state leaf or intermediate field is introduced. Lowering resources
are measured above. GPU runtime, VRAM, executable E41 and CPU-WRF trajectory
D6 remain unknown/pending and this review makes no speed/fidelity claims.

RMP00 (candidate predates RAM wrapper) and RMP02 (missing c40 target file) were
setup-only failures. The corrected run used the existing c62 target with the
exact producer target SHA; no GPU query and no repeated successful caller test.
The failed attempt is retained. Snapshot/test scratch are removed after sealing;
no disposable JAX cache was created. One critic, no review of a review.
