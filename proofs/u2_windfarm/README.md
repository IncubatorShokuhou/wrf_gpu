# Fitch qualification packet

`MERGE_TRAIN_HANDOFF.json` records the subsequent P3/mp22/FDDA/Milbrandt
rebase. Approved Fitch sources remain byte-identical; main's carry/restart/
catalog order and FDDA handling are preserved. The manager transferred final
composed validation to s2-integrate during that rebase. The measured checks
below remain historical evidence for their recorded source trees.

The final main rebase and manager-requested composition checks are sealed in
`REBASE_PACKET.json`: 498 current targeted checks pass, with 64 prior checks
carried under the recorded reuse scope; the official Swiss release identity
remains exactly `2bf12cb8`. Main carry/restart/catalog order precedes the Fitch
additions. The original packets below remain unchanged historical evidence.

The review correction is sealed in `FIX_PACKET.json`. It supersedes the
kernel-association and affected-component/cost claims of the historical
`PROOF_PACKET.json`, which is preserved unchanged. Only `turbine_area` changed
in product source. The fix preserves WRF's left-associated ELSE sum; five
independent asymmetric scalar witnesses pass BYTE and the old grouping mutant
fails all five. The corrected component passes all 23 affected tests and the
original 31,968-value diagnostic BYTE fixture. See `UPSTREAM_WRF_BUGS.md` for
expanded-geometry residuals; these fixtures do not establish universal BYTE
parity. No product compiler flags or tolerances changed.

Scope: opt-in `windfarm_opt=1`, MYNN PBL/surface layer 5/5, every-step PBL,
WRF turbine files and one-based ij or WRF-projected lat/lon placement. MAV=2
remains refused. User instructions: `docs/wind-farms.md`.

`oracle.py` compiles the three unchanged pristine Fitch routines with a small
caller wrapper. The tracked 72-scenario REAL4 fixture covers cut-in/out,
standing coefficients, exact/interior curve knots, shear, colocated turbines,
different table lengths, the nonpositive Ct−Cp branch and absent rotor layers.
The predeclared normal-compiled bounds are rtol 3e-6, source atol 2e-7, and
POWER atol 0.25 W. The literal no-FMA path is 31,968/31,968 output values BYTE.
Initialization sensitivity and compiled contraction deviations are recorded in
`UPSTREAM_WRF_BUGS.md` and `receipts/contraction.json`; no tolerance changed.

The genuine mutation control removes WRF's power1/power2 energy normalization
in memory. The independent pristine oracle rejects it. `mutant.py` reproduces
that control without changing product files.

Tests cover file validation, raw WRF map placement (reusing original llxy
savepoints), caller wind/QKE ownership, both CLI bindings and admission,
doctor density advice/input hashing, POWER history, pickle/NetCDF/checkpoint
and native-store restart. Rehashed missing-group/configuration corruptions and
changed turbine tables are refused. Daily replay selects the continuous
full-carry stepper and retains POWER/global cadence across segment boundaries.
`namelist_oracle.py` independently checks WRF array defaults with a Fortran
NAMELIST reader: five supported scalar/vector/repeat/null forms agree exactly.
Omitted `windfarm_opt` domains remain zero; indexed assignment retains the
shared parser's explicit refusal. The measured Swiss/PROD/wind-farm d01 bound
options are identical before/after this host-only correction.

The full Swiss CPU smoke uses unmodified real meteorology/grid with three
explicitly synthetic turbines (42×42×44, dt=18 s). `coupled_smoke.py` executes
two complete production steps, retaining zero nonfinite/guard events, and
continues the second step from the native restart store. Every carry leaf also
round-trips byte-exact through pickle and NetCDF. FAST_DEFAULTS=0 plus the
accepted CPU Thomas shim is a CPU integration test, not GPU execution or
original-WRF coupled-forecast parity. The final source receipt identifies the
exact measured commit.

The official portability-aware release probe confirms unchanged OFF graphs:
Swiss `2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`
and PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
`release_identity.py` delegates to `proofs/s1_port/release_identity.py`, changing
only its temporary directory into this lane. PROD is a WN2 iteration identity
check; it supplies no final forecast-fidelity claim.

Performance/memory: OFF graphs are exact. ON adds one REAL surface POWER leaf
(4×ny×nx bytes). The sparse sequential turbine loop uses compact curve
constants, preserves colocated turbine order and avoids an nt×nz×ny×nx
temporary. On a CPU 20×40×80 component probe, 1/10/100 turbines compile in
0.314/0.552/0.865 s with 1,896/2,584/2,584 bytes of reported temporary storage.
Warm CPU medians 0.207/0.096/0.412 ms were collected under contention and do not
establish scaling or GPU speed. These are component figures, not full-model
peak memory. Including actual mass-wind and geopotential/dz preparation, the
100-turbine hook compiles in 0.455 s, takes 0.676 ms median on CPU and reports
1,039,384 bytes of temporary storage (about 1.037 MB of dense preparation plus
2,584 bytes for the sparse loop), with zero optimized f64 type mentions. GPU
fusion, launch/traffic cost, cold readiness and
E41 remain unmeasured under the wave's CPU-only rule.

CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Full CPU-WRF
D6 and release performance/resource gates remain binding. Experiment receipts
and commands are append-logged from WF01 onward in the main EXPERIMENTS.tsv; selected
small receipts are committed here. Temporary snapshots, builds and caches are
removed after their final dependent arm.

Original admission and receipt hashes are in `PROOF_PACKET.json`. The rebased
composition passed all 379 selected checks (378 in WF21 and the corrected stale
RUC fixture in WF22). Final Swiss/PROD native-facade graphs are again exact in
WF23/WF24. The kernel/adapter sources and measured root bindings are unchanged;
the original packet records the numerical-evidence reuse and the inactive
upstream urban path. The review fix changes kernel association and has its own
affected gates in `FIX_PACKET.json`; caller/IO/restart/OFF evidence is carried
for unchanged source, without claiming a new full coupled run. The corrected
100-turbine full hook measured 0.275 s compile and 0.339 ms CPU median, with
unchanged 1,039,384-byte temporary storage, 12,800-byte POWER and zero f64 type
mentions. Timings were collected under contention and imply no speedup claim.
