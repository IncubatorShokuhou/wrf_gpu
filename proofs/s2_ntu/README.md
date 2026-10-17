# NTU microphysics, mp_physics=56

This is an opt-in experimental port of pristine WRF 4.7.1
`phys/module_mp_ntu.F`, SHA256
`c864483fa572d42ab1dc51cc538f8062dc2f582369f597261e8c850c49ce34dd`.
The caller is `module_microphysics_driver.F:1586`, with original Registry
moist/scalar fields and `ccnty=1..4` (default 2). No older GPU port is a
scientific reference. The final gate ledger is `QUALIFICATION.json`.

## Independent numerical evidence

- `fixtures/columns_{r4,r8}`: 12 warm, cold, mixed, evaporating and ice
  supersaturated columns, two evolving calls, all prognostic species active
  across the corpus. All 8,568 public values per precision match the original
  no-FMA WRF oracle byte for byte through the literal executor.
- `fixtures/scalars_*`: all six SOLVE_AFA species operators, ICENU,
  sedimentation and both adaptive timestep operators.
- `fixtures/activation_*`: 128 calls per precision, all four CCNTY banks,
  activation/held regimes and original DOUBLE Newton solution.
- `fixtures/initialization_*`: 2,880 values per precision, literal BYTE;
  the operational NumPy cold initializer differs by at most 2 ULP in REAL4
  and 1 ULP in REAL8. It uses actual dry specific volume and geopotential.
- `compiler_controls/`: original no-FMA/native-FMA compilers plus each input
  field's independent +/-1 ULP perturbation, fixed before port comparison.
  The baseline oracle is byte-identical to the frozen fixture.
- `COMPILED_DEVIATIONS.json`: compiled CPU Pallas has 8,568 finite values,
  with **313 outside the fixed 134-control enclosure**. Maximum mass-field
  excess is 8.43e-9 kg/kg; theta 1.22e-4 K; RAINNC 1.49e-8 mm. This is a
  disclosed experimental deviation, not an enclosure PASS or D6 result.
  `compare_compiled.py` and `test_ntu_native_evidence.py` reproduce the list
  and bind all numerical dependencies by AST hashes.

Original uninitialized-read sensitivity, the original negative cube root
discarded by an explicit source MAX, and NTU's public precipitation partition
are documented with source witnesses in `UPSTREAM_WRF_BUGS.md`. The port
uses the standing compiler-zero policy and introduces no repair clamp.

## Integration and regression evidence

The option binds through both pipelines and the honour check, transports all
seven moist and 21 scalar fields, consumes available original root-boundary
names, preserves original history metadata/accumulation, and round-trips the
complete carry. Missing, malformed or nonfinite NTU restart fields fail
closed. Actual bound aerosol providers and CCNTY are part of the AOT key.
Live nests are refused until separately qualified.

`tests/v040/test_ntu_wiring.py` covers these contracts, all three writer modes,
the original raw precipitation caller and actual table-key collisions.
The required JAX `test_microphysics_operational_runs_and_mutates[56]` passes
separately from the optional NumPy coupling evidence; see
`CPU_COMPILATION.json`. Shared suite receipts, default identities, clean
archive import, the physical mutant and coupled smoke are sealed in the
final qualification ledger. Numerical validation is reused only under exact
source or numerical AST identity.

`coupled_smoke.py` retains real Swiss inputs, complete RK and other physics,
two 18-second steps and full-carry restart. Its NTU call uses the same
column equations through the explicitly labelled NumPy backend; it is a CPU
coupling/finite smoke, not JAX/GPU or CPU-WRF forecast accuracy. CPU-only
cache releases and existing 128-column RRTMG tiles bound harness memory;
the M9 caller's explicit tile setting is overridden as well as the solver
environment. Guard events need zero counts or a matched control.

The complete pre-mask capture changes the interpretation of the initial
samples. The default control has zero events; NTU has 93 qi and 127 Ni
post-RK visits in the legacy CPU path, reaching -1.3227e-10 kg/kg and
-954.597/kg (about 1.5% of the affected column maxima). **These are not
round-off.** Selecting the existing original-native PD path retains the
same large class. Independent pristine `sixth_order_diffusion` on the
captured full-grid inputs produces the source-driven negative class: its
RK1 tendency uses dt/3 and is reused by the final dt update. The replay is
`replay_diffusion_f2.py`, with sealed inputs in `f2_source_diffusion/`.
`QUALIFICATION.json` retains the failed 1e-12 round-off predicate separately
from the **accepted case-scoped source-diffusion F2 known issue** (manager
decision 2026-10-14 22:44:27Z). Full original RK-chain comparison is not
required for this disposition. The captured settings are `diff_6th_opt=2`,
factor 0.12, slope option 0, dt=18 s and CCNTY=2; the scope is this two-step,
44-level single-domain CPU FAST0 continuation, including the native PD path
bisect. No model equation, WRF diffusion coefficient, input, tolerance or clip
was changed. This does not establish global F2 or complete original-CPU/GPU
coupled forecast qualification. See the
[known-issue disclosure](../../KNOWN_ISSUES.md#ntu56-development-case-scoped-f2-source-diffusion-undershoots).

The full CPU smoke uses FAST0: FAST1 shortwave exceeds the 6 GiB harness
cap before reaching NTU. New NTU-owned fields remain REAL4; pre-existing
shared qh/Nh/hail leaves follow the FAST0 wider State policy and their dtypes
are recorded. The NTU equations themselves always enter at REAL4 through
the operational adapter. This is not a FAST1 or native forecast test.

## Performance and memory limits

`NATIVE_PREFLIGHT.json` is the exact 40-output production projection at 44
physical levels, a 64-level register tile, four warps and one Pallas program
per column. CPU CUDA/sm120 TTIR lowering passed in 7.51 s, with zero fp64
arithmetic outside original WRF DOUBLE islands. No GPU executable was
compiled: GPU launch count, spills, E41, VRAM, speed and D6 remain unknown.

The required CPU JAX singleton takes 4.49 s to lower, 79.83 s to compile at
LLVM opt0 and 11.48 s to execute. Peak RSS is 6.87 GiB, zero swaps; it passed
under a manager-authorized 12 GiB exception after the 6 GiB cap rejected it.
Microsteps, convergence, sedimentation, columns and levels are not
Python-unrolled; the shared generated physics graph is large. The separate
CPU Pallas numerical arm spanned host hibernation: **all its timings are
invalid (E60)**; numerical observations and static buffers remain valid.

Selected NTU carry adds 17 REAL volume arrays and four surface arrays,
about 5.06 MiB on 42x42x44 before transport scratch. Default options allocate
none of these arrays. No host transfers are added to the GPU timestep loop.
Full GPU and original-CPU coupled forecast qualification remain pending.
