# Experimental EEPS / KEPS qualification

The full REAL32 EEPS (bl=16) and KEPS (bl=17) ports use unchanged WRF 4.7.1
bodies as their independent reference. Manager-authorized compiler options are
`-finit-real=zero -finit-integer=0 -finit-logical=false`, FP traps off.
Four undefined upstream reads are mirrored explicitly at the documented source
sites. Alternate compiler initialization materially changes both schemes;
see `initialization_sensitivity.json` and `docs/EEPS_KEPS.md`. Neither option
claims compiler-independent WRF semantics. WRF 4.7.1 has no bl=6 package.

`full_zero.json` has 60 carried full-driver records: 30 per scheme, 16/44 levels,
stable/unstable/neutral/stratocumulus/mixed-ice/restart inputs, five steps each.
`build_full_oracle.py` builds private unchanged copies, initializes wrapper
operands explicitly, and records source/compiler/driver/input hashes.
`full_compare.py` applies the frozen 16 REAL32 ulp state/diagnostic gate and
propagated input rounding for cancellation-sensitive tendency and flux outputs.
Both arms must be finite; integer KPBL is exact. All 30 records per scheme pass
with a fresh CPU target using SSE4_2 and disabled algebraic reassociation.
Native CPU ISA and GPU precision remain unqualified. GPU/coupled-forecast D6
qualification is pending even when short CPU coupling checks pass.

`literal_byte_check.py --source-real32` executes these same port function bodies
through the proof-only scalar REAL4/libm evaluator in `literal_real32.py`.
All 1,140 field comparisons (26,690 values) are BYTE against unchanged WRF.
This isolates source association from XLA transcendental approximations and
subnormal flushing. It is separate from the compiled JAX gate above; the host
evaluator is never imported by the forecast. The default eager-XLA audit retains
its measured residuals and is not described as byte-identical.

`operators_real32.json` separately covers 18 numerical helper profiles and
324 output fields; it is a helper reference, not a forecast. `manifest.yaml`
describes those local operands. `full_manifest.yaml` describes the full driver.
The genuine TPE deletion and nonfinite gate mutants are tested in
`tests/test_v040_keps_operators.py`. Carry/schema/boundary deletion and exact
restart checks are in `tests/test_v040_epsilon_wiring.py`.

Operational selection is explicit and currently requires sfclay=1, bldt=0,
flux-form scalar transport and raw RK physics rates. EEPS transports pek_adv /
pep_adv; KEPS transports tke_adv / diss_adv / tpe_adv even though WRF disables
feedback of those fields into its KEPS closure. New physical/transport/held
fields and the call counter are REAL32/int32 and versioned on all restart
formats, including native synchronized RestartStore admission before device
allocation. Both pipelines bind/honour the selector. No-LSM and Noah-MP are the
qualified land combinations; others fail the honour check. The Noah-MP surface packet
uses the shared optional interface from s2-temf commit84020b732; it performs
one surface solve and retains actual RMOL/WSPD/BR/FM/FHH rather than reconstructing
them from another scheme. History reads held fields, without another PBL solve.

Reproduce from an immutable checkout (current CPU grant/QUIET rules apply):

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src \
 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_cpu_max_isa=SSE4_2 --xla_disable_hlo_passes=algsimp' \
 taskset -c 12,28 nice -n 19 timeout 600 python proofs/s2_keps/full_compare.py \
 proofs/s2_keps/full_zero.json <USER_HOME>/wrf_gpu2_lanes/s2-keps/full_compare.json
```

`smoke.py` loads the real Swiss 42x42x44 CLI case and can execute three steps
with O1_EXEC_STEPS=3, native FAST1 CPU fallback, GPUWRF_CARRY_DONATE=0 and census.
Donation is disabled only in this diagnostic harness, which retains the initial
carry to report deltas and bypasses the public runner's donation preparation.
Such checks establish integration,
finite values and guard counts, not independent coupled forecast fidelity.
`cost.py` reports full-column CPU compile/warm/temporary memory; these numbers
do not estimate GPU launches or roofline performance. `archive_check.py` checks
tracked fixtures and imports from a clean archive. No new runtime asset is used.
Default-model jaxpr identity is checked separately against the full Swiss
2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a hash.

The third native CPU KEPS step reports one repaired QC value at the north
boundary, -1.2946662e-31 kg/kg. The actual interior source point is (k,y,x) =
(15,40,8), with positive north-face transport. Original WRF `flow_dep_bdy`
copies this value to (15,41,8). The actual full QC PD operands are tracked in
`fixtures/pd_qc_step3.npz`, with custody metadata in `pd_qc_event.json`.
`pd_oracle_check.py` uses the existing unchanged WRF `advect_scalar_pd` and
`rk_update_scalar` wrappers on these operands. WRF without FMA returns
2.0436820e-32 kg/kg; contracting WRF returns -4.7719314e-32 kg/kg. WRF reproduces
the negative sign. The candidate's error is 2.20 times this compiler spread,
within the frozen eight-times C05 envelope; full-field max/RMS errors pass too.
The additional strict one-times point-interval audit remains **OPEN**. Its excess
is disclosed under the standing FMA policy; full-forecast D6 decides admission
beyond this experimental CPU qualification. The raw strict-zero smoke and
strict-interval receipts are retained. Both diagnostics reproduce the OFF
arm's event counts, locations, values and maximum State deltas exactly.
No model clamp was added.
`pd_qc_reference.json` retains the original-source hashes and results.
