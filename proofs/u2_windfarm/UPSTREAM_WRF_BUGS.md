# Pristine Fitch bounds and compiler behavior

Pristine source SHA256: `83247add74780608e798bb62424533b18abf31f3da5aba6374de6fe2550060c3`.
All three computational subroutines are unchanged; only their module/MPI/file
initialization dependencies are supplied by the standalone wrapper.

`module_wind_fitch.F:158–179` leaves `z1` unassigned for a hub below the first
mass level, or `ktop` unassigned when no mass level brackets the hub above.
Subsequent turbines may also inherit the preceding turbine's indices. These
geometries are outside this port's admitted scope and refused at input loading;
the port makes no fidelity claim for them. Full rotors and bracketed hubs are
required. No pristine body is edited or physical repair applied.

The oracle now uses compiler-defined zero initialization (`-finit-real=zero`,
`-finit-integer=0`, `-finit-logical=false`) and the production default of disabled
FP traps. The 72 admitted scenarios are byte-identical to the original fixture
and to poison initialization (signaling NaN, integer −77777, logical true).
No uninitialized-read dependence occurs on the admitted test inputs.

The literal JAX REAL4 path is **31,968/31,968 values BYTE** vs pristine: DU/DV/QKE
10,368 each and POWER 864. CPU diagnostic flags are
`--xla_cpu_enable_fast_math=false --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp`.
These are proof settings only; product CPU/GPU ISA settings remain unchanged.
Reproduce with `FITCH_REQUIRE_BYTE=1` and those `XLA_FLAGS` while selecting
`tests/test_u2_windfarm.py::test_pristine_real32_dragforce_all_regimes` on CPU.

The normal compiled JAX path has the following absolute differences from the
literal reference. The independent WRF spread uses original Fortran `-O0`
versus `-O3 -march=native -ffp-contract=fast`, with identical initialized inputs.

| Field | JAX maximum difference | WRF maximum compiler spread | Largest per-value excess over WRF spread |
|---|---:|---:|---:|
| DU (m/s²) | 3.7253e-9 | 4.6566e-9 | 9.3132e-10 (441 values) |
| DV (m/s²) | 4.6566e-10 | 4.6566e-10 | 2.3283e-10 (12 values) |
| QKE (m²/s²) | 2.8610e-6 | 2.8610e-6 | 1.9073e-6 (112 values) |
| POWER (W) | 2 | 2 | 1 (24 values) |

Each field's maximum JAX difference is within WRF's maximum compiler spread.
The per-value excesses are explicitly disclosed REAL4 compiler/intrinsic
deviations under the standing policy. The original component tolerances remain
unchanged; full CPU-WRF D6 and GPU/coupled forecast qualification remain pending.
No input floor, tolerance relaxation or product compiler flag is used.

The independent review found an additional source-order defect in the
rotor-center ELSE branch: `fd+fc` grouped the four terms into two pairs, while
WRF adds them from left to right. The corrected product uses `(fd+c1)+c2`;
the two same-sign branches and the final density multiplication are unchanged.
Five asymmetric scalar witnesses supplied by the critic's unchanged pristine
REAL4 build are tracked in `area_order_witnesses.json`, with source and review
receipt hashes. All five are BYTE with diagnostic flags; restoring the old
grouping in memory fails all five (`area_order.py --mutant`). WF25/WF26 also
recheck the normal-compiled frozen bounds and the original 31,968-value BYTE
fixture on the corrected product.

BYTE statements above describe these fixtures, not arbitrary geometries.
The critic's expanded 4,096 asymmetric vector cuts retain 501 BYTE mismatches
with literal association (versus 1,138 with the old grouping). Its 72-case
hub=81 m driver retains DU/DV/QKE mismatches of 366/12/114 values, with maximum
absolute differences 1.862645149e-9 / 2.328306437e-10 / 1.430511475e-6;
POWER remains BYTE. These are independent review measurements RV07/RV08,
carried as compiler/intrinsic residual disclosures, not remeasured final-tree
claims. Fixing association does not remove every intrinsic/compiler deviation.
Review evidence: `<USER_HOME>/wrf_gpu2_lanes/rv-u2-windfarm/VERDICT.md` and
`RV07-area.json`, `RV08-rotor.json`, `RV10-scalar.json` in that directory.

Receipts: `WF09-nofma.json` and `WF10-contraction/receipt.json` in the lane
artifact directory, with committed copies in `receipts/`. Reproduce the
initialization/contraction comparison with `contraction.py --out LANE_DIR`.
