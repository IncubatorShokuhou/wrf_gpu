# QNSE surface/PBL and ARW EDMF — experimental CPU oracle qualification

The port implements pristine WRF V4.7.1 ARW QNSESFC, QNSEPBL and the separately
called MFSHCONVPBL (`KRR=2`, `KRRL=1`, `KRRI=0`, `OMIXUV=False`, `PIMPL=0`).
Select `bl_pbl_physics=4` with `sf_sfclay_physics=4`; `mfshconv=0/1` selects
mass flux, with WRF's default 1 and reset to 0 outside QNSE in both loaders.
Coupling supports no LSM and Noah-MP. Slab (`sf_surface_physics=1`),
Noah-classic (2) and Pleim-Xiu (7) are refused at API and namelist admission:
their physical HFX/QFX handoff and all caller operands need qualification;
WRF slab also supplies CHKLOWQ=MAVAIL rather than the adapter's current 1.
The surface/PBL compiled input-ULP gates pass. EDMF has a disclosed compiled
rounding deviation under the updated standing wave policy. Swiss MF0 and MF1
three-step CPU smokes are complete. MF1 uses the declared legacy CPU emitter
configuration below; production compiler defaults are unchanged.
No GPU or coupled-forecast qualification is claimed.

`oracle.json` contains 15 paired surface/PBL columns in REAL4 and REAL8.
`mf_oracle.json` adds the actual ARW mass-flux call before QNSEPBL. Four heap
perturbations and explicit logical-initialization controls produced identical
original-WRF outputs: 24/30 precision/regime pairs had active mass flux and
4/30 had cloud. These are independent, constructed WRF differential columns,
not station or forecast validation. Unmodified WRF source hashes and driver
hashes are stored with each fixture. The exact initializer tables come from
unmodified QNSESFCINIT; their source/driver/asset hashes are in surface_tables.json.

The manager's compiler-rounding decision replaces obsolete absolute floors:

1. `real4_literal.py` evaluates the SAME model functions with rounded REAL4
   arithmetic and the same scalar libm ABI as the pristine gfortran build.
   Qualification compares actual bytes, including signed zero. No production
   code imports this evaluator. Q18 verified all 45 columns BYTE after the
   final static-level refactor, including signed zeros.
2. Surface and PBL compiled outputs stay in independent propagation enclosures made
   exclusively from inputs, source expressions and arithmetic error contracts.
   The bound construction never reads reference or compiled outputs. Infinite
   bounds are refused. `enclosures.json` freezes all 30 finite enclosures;
   Q22 checked every reference and compiled output inside them. There are no
   fitted absolute floors, and no D6/station tolerance is changed.
3. EDMF's attempted affine enclosure diverged at a cloud transition and was
   REFUSED. `compiler_rounding.json` instead records the independently measured
   pristine WRF compiler controls and the **raw compiled-spread gate FAIL**.
   This is disclosed under `_WAVE_S1_COMMON.md`'s updated contraction policy;
   it is not described as a successful input-ULP proof. The compiled MF test
   checks byte identity to the disclosed CPU artifact, a regression test only.
   Literal BYTE remains the source fidelity gate; coupled D6 remains binding.

`ulp_bounds.py` evaluates outward intervals from +/- one REAL4 input ULP
(nonzero continuous inputs; prescribed zeros and discrete controls are exact).
Each arithmetic result includes 0.5 ULP; division allows 2+0.5, real power
4+0.5, exp 2+0.5, log 1+0.5, and correctly-rounded sqrt 0+0.5. CUDA13.1
mathematical-function Tables47/48 supply the library budgets; an executed GPU
backend/intrinsic contract is not qualified by this CPU packet. Selection
unions both reachable branches. Surface interpolation enumerates every
reachable integer bin jointly with its fractional coordinate, avoiding false
independent table jumps. Integer powers use rounded multiplies. The normal,
finite identity sqrt(round(x*x))=abs(x) requires correctly-rounded sqrt.

For mass flux, `affine_bounds.py` retains original input symbols and shared
rounding residuals: c + sum(a_i*e_i) + [-r,r], |e_i|<=1. Products retain linear
terms and add the full width-product remainder; unary functions add half the
supremum second derivative times width squared. Each reused residual is lifted
once into a shared symbol. The representation retains at most 4*nz symbols;
discarded coefficients enter r with their full outward L1 norm. Compression
only widens an enclosure and does not define an accepted error floor. Positive
binary64 sums use Gamma_n widening. A cold or wholly unsaturated thermodynamic
state is a certified fixed point of all 20 source iterations; other states run
the unchanged iteration source. Dead entrainment results are pruned only when
the caller's entire enclosure has before=False and masks every returned value.
These MF proof calculations failed to establish finite enclosures and are
retained as rejected research evidence, never as an acceptance tolerance.

The WRF REAL4 controls use unmodified bodies, compiler-defined zero initialization,
`-O2 -march=native -fno-tree-vectorize -fno-tree-slp-vectorize`, and respectively
`-ffp-contract=fast/off`. Disassembly counted 248 scalar REAL4 FMA instructions
versus zero. Both controls start from the same original driver inputs. The MF
comparison includes upstream surface arithmetic changes; it is a caller-chain
compiler control, not an isolated identical-intermediate-input claim.
Across the 15 MF cases, 130 values exceed the per-cell WRF spread. Maximum
excesses are RTHBLTEN 1.017469913e-6 K/s, RQVBLTEN 9.299583326e-11 kg/kg/s,
mass flux 3.099441528e-6 kg/m2/s, RC_MF 6.490154192e-9 kg/kg, and updraft
virtual theta 1.220703125e-4 K. All fields, cells, errors and spreads are in
the packet. No error-derived floor is used. GPU arithmetic remains unmeasured.
Initialization sensitivity is documented in [UPSTREAM_WRF_BUGS.md](UPSTREAM_WRF_BUGS.md).

WRF caller details retained include dynamic P_PHY/P8W and hypsometric RHO,
dry theta and EXNER, incoming TKE for coefficients, land HFX-based THZ0,
first-step land QS, additive five PBL tendencies, preserved EXCH interface slot
zero, signed Coriolis floor, and source scratch-variable overwrite order in
entrainment. WTHV/LM_BL89 do not enter active QNSEPBL expressions. Runtime
surface exchange is solved once, held through the land call, and consumed with
updated land HFX/LH by PBL. Restart version 1 requires all QNSE memory in pickle,
NetCDF, M7 checkpoint and native synchronized RestartStore, including public
save/read validation. Q17: 41 restart/admission checks passed; both loader seams
and mandatory pairing were exercised.

QNSE also requires the existing six REAL4 history arrays (24 bytes/surface
cell, already enabled by release FAST1). This prevents the output writer from
re-solving MYNN when the global cache flag is disabled. Non-Noah-MP paths hold
QNSE's own T2/Q2/U10/V10/PSFC and updated land LH; Noah-MP writes its existing
blended diagnostics. All four restart routes require this cache and match
QNSE profile dimensions to the actual State. Q27: 75 checks passed, including
20 bad-cache/domain writes and four deliberately dropped-cache reads.

Three Pallas column kernels keep levels, surface sweeps and the 20 saturation
iterations inside the kernel. Q16 CPU interpretation and CPU-to-sm120 lowering
passed for all three with zero f64 types. Contended CPU compile plus first call
(nz=16, one column) was surface 16.465s, PBL 118.318s, MF 494.534s. Native TTIR
sizes were 315426/1042279/2858510 characters. Q19's 15-column CPU batches were
finite for all 45 cases; compile plus first call was 9.581/41.960/205.605s on
the updated core allocation. These are CPU diagnostics, not GPU
speed, SASS/E41, peak VRAM, or readiness qualification.

The portable CPU dispatcher now evaluates the same register-level equations
directly, retaining all 20 thermodynamic iterations. Explicit Pallas
interpretation remains available for qualification. Q30 verified all three
native GPU kernel jaxprs and constants byte-identical across this change,
with zero f64 types. The 62-test column suite passes with both CPU paths;
the portable MF batch is byte-identical to the disclosed interpreter artifact.
See [VALIDATION_REUSE.json](VALIDATION_REUSE.json) for the exact proof chain.
Contended CPU batch compile/first-call was MF 93.26s versus interpreter
107.83s, PBL 26.37s versus 38.46s in that suite. These are diagnostic timings.
The full Swiss MF1 interpreter arm timed out at 2700s before any step
(sampled RSS about 26 GiB); the portable single-core arm also timed out at
3600s. A fresh eight-logical-core process compiled the model but did not return
its first step after over twelve further minutes. Its resident high-water mark
was 49.69 GiB, with a later 18.37 GiB swapped-memory sample. A process-local
native-stack probe identified hot nested scalar `select_n` CPU fusions, and
MLIR compiler-context cleanup with swap I/O. These default-emitter arms remain
incomplete, without a numerical verdict. Native GPU compile, stack/spills and VRAM remain
unqualified; CPU interpreter expansion is not GPU code size.

The completed MF1 CPU smoke selects
`XLA_FLAGS=--xla_cpu_use_fusion_emitters=false` with the same model functions,
real inputs, FAST1 and CENSUS1. All three carries are finite, all QNSE leaves are
REAL4, and guard totals are 0/0/0. First-step wall including compilation is
585.46s; warm steps are 13.90/11.63s. PBL and surface outputs still pass the
independent input-ULP enclosures under this CPU setting. MF differs from the
frozen default-emitter regression artifact: the separate
[compiler_rounding_legacy_cpu.json](compiler_rounding_legacy_cpu.json) records
136 raw pristine-compiler-spread excesses, with finite candidate/reference
checks and byte-equal no-FMA inputs. Maximum RTH/RQV/RC_MF excesses are
1.01747e-6 K/s, 1.40062e-10 kg/kg/s and 3.68163e-9 kg/kg. This remains a
disclosed compiled deviation; the original 130-cell default-emitter artifact,
all numerical tolerances and D6 are preserved.

[COUPLED_SMOKE.json](COUPLED_SMOKE.json) holds exact Swiss input hashes and the
three-step MF0 and release-control receipts. Both finish with finite carry and
guard totals 0/0/2: `dynamics.qc` has zero nonfinite, one out-of-range and one
repair at k=15/y=41/x=8, ring 0. This matches the standing policy's pre-existing
field/count/location class; the original raw guard FAIL is preserved. This is
not an amplitude-based WRF fidelity claim or a global F2 waiver. Portable MF0
first-step wall was 1126.72s including compilation, followed by 13.67/12.41s.
The final EEPS/KEPS-composed host gate passed 315 tests, including
PhysicsStepSpec contracts and the legacy microphysics fail-closed pins;
release identity remains exact `2bf12cb8`. Across the final main rebase, all
194 initial carry leaves and the full 294,298,886-character MF1 CPU StableHLO
are identical (SHA256 `f54dbd090b1ade06e841d3ed9f61fcb991f7c0e9918d0bef66be6953da8bb120`).
The completed MF1 numerical receipt therefore carries forward unchanged.

Held memory is nine 2-D and three 3-D REAL4 fields (36 bytes/surface cell plus
12 bytes/model cell), with existing State QKE=2*TKE. The shared REAL4 lookup
tables occupy 80008 bytes. Production MF exports only RTH/RQV/RC_MF (three
profiles instead of 19) and skips unconsumed full PLM diagnostic work. The full
19 outputs remain available to the oracle path. GPU launch/traffic/peak and
forecast costs remain unmeasured. Q19's updated native-facade release probe
returned the exact full Swiss 2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a.

Regenerate fixtures only from a staged copy in the lane directory, with the
QUIET guard, JAX_PLATFORMS=cpu, GPUWRF_JAX_CACHE=0, assigned affinity/nice and a
timeout. Tools: build_oracle.py, build_mf_oracle.py, build_surface_tables.py.
Run tests/test_v040_qnse.py and tests/test_v040_qnse_coupling.py with
--basetemp <USER_HOME>/wrf_gpu2_lanes/s2-qnse/pytest. Do not replace fixtures
without recording source, compiler, caller and numeric-array identity.
