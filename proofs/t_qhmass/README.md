# Active hail in WRF moist-mass loading

The original counterexample gives PSFC 95000 Pa instead of 95090 Pa when
`qh=0.001`, dry column mass is 90000 Pa and model-top pressure is 5000 Pa.
The fix includes hail at all three affected caller sites, using one shared
Registry-derived mass selector. Number concentrations, predicted volumes,
aerosols and rime partitions are scalar members and do not add air loading.

| Site | Pristine WRF definition and caller |
| --- | --- |
| Acoustic stage | `module_em.F:171` passes the full moist array to `calc_cq`; `module_big_step_utilities_em.F:856–870` sums every active member at adjacent levels. `pg_buoy_w:2474–2497` consumes this loading. |
| Physics pressure | `module_first_rk_step_part1.F:208–224` passes moist, MUT, C1H/C2H and DNW to `phy_prep`; `module_big_step_utilities_em.F:4940–4973` integrates every active mass species downward. |
| History PSFC | The surface driver receives `grid%p_hyd_w` and takes its bottom interface (`module_surface_driver.F:1988`), so the same all-moist integral applies. |

`Registry.EM_COMMON` selects separate QHAIL for NSSL 17/18 and the other hail
mass packages. NSSL 22 disables hail (`module_check_a_mundo.F:3397–3402`).
Thompson 38 predicts a combined graupel/hail category in QGRAUP, with scalar
number/volume moments; it has no separate QHAIL mass. Legacy NSSL 17 normalizes
to 18 (`module_check_a_mundo.F:3406–3415`).

The six core State slots remain in their original accumulation order, including
their existing cast placement. Absent optional mass fields generate no device
operations. The native hydrostatic sweep accepts all optional mass inputs
directly, retaining one column kernel and avoiding a packed/full-grid sum.
The central Registry and new helper belong to the existing AOT source import
closure; there are no new external baked inputs.

The oracle extracts unchanged `calc_cq` and hydrostatic `phy_prep` loops into a
private REAL4 shared library. It preserves WRF's unused first moist index,
`PARAM_FIRST_SCALAR=2`, the column bounds and full caller argument mapping.
Each build records pristine source/body hashes and compiler arguments.
The literal REAL4 profiles and CUDA round-to-nearest instruction semantics
must match byte for byte. The CPU Pallas interpreter permits contraction;
the byte test therefore emulates the actual CUDA `.rn.f32` operations with
the established exact-rounding shim. The retained aggregate-first acoustic
sum is compared with WRF's species-interleaved face sum using a pre-derived
floating-point sum bound. The original PSFC counterexample retains its
1e-9 Pa absolute bound. See [oracle_manifest.yaml](oracle_manifest.yaml).

`coupled.py` loads the real Swiss CLI case, changes only the microphysics
namelist selection and runs two complete CPU steps with the established
FAST0/128-column/cache-release configuration. At each loading seam it checks
the actual operands against pristine WRF. These are caller/continuation checks,
not a full CPU-WRF forecast trajectory comparison. `program_identity.py` runs
the established canonical release probe with the native facade held at trace
time; parent comparisons are regression evidence only.

Measured CPU frontend evidence: six-, seven- and eight-mass native kernel
jaxprs are byte identical to the parent; nine-mass lowering also succeeds.
Every variant has one Pallas call and zero runtime fp64 arithmetic. Default
Swiss `f672a573` and PROD `6fc715a2` remain exact; full NSSL 22 and Thompson 38
programs match the parent. The changed hail path adds a mass load/add per cell;
GPU launch, temporary-memory and timing consequences require the composed
GPU executable. No GPU timing, E41, S3 or forecast fidelity claim comes from
these CPU checks.

The changed-source GPU/D6 re-run set in `v-prep/MATRIX.json` is c07 (mp9),
c12 (17), c13 (18), c14 (19), c17 (24), c18 (26), c64 (7), c88 (56).
NSSL 22 and Thompson 38 can carry forward this fix's unchanged-program proof.
Final receipts and any outstanding qualifications are recorded in STATUS.json.

The sealed CPU gate has eight focused tests and 344 shared checks passing,
three genuine deletion-mutant assertion failures, and 45 pristine loading
observations over three two-step caller runs. NSSL 17/18's active carry check
uses original CPU-WRF QHAIL/QNHAIL from hour 12 on the byte-verified Swiss grid:
1,815 hail cells remain active after microphysics, and the interior field
reaches the next RK caller byte unchanged. This controlled input check does not claim forecast trajectory
parity. NSSL 22 and Thompson 38 use unmodified Swiss inputs. Every completed
step has zero non-finite values and guard events.

The shared ShinHong QKE-mutation assertion also fails after restoring only the
five pre-fix functions; it is retained as an inherited failure. A source-only
follow-up identifies the separate direct writer PSFC fallback's missing
optional masses; this writer path is outside the three owned model sites and
was reported to the manager. No writer code was changed.
