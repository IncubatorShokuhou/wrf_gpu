# Milbrandt-Yau mp=9 qualification

This is CPU-oracle qualification of an opt-in single-domain scheme. GPU
execution, E41, GPU speed/VRAM and forecast D6 remain pending.

The pristine reference is WRF `phys/module_mp_milbrandt2mom.F`, SHA256
`76d11c00976f3c7dac30ff6ebb3efac1698bf43c13131b3aae43e464f22d23e5`.
The lane build copies unmodified module bodies. `oracle/oracle.F90` calls the
main boundary with WRF wrapper flags; `oracle/wrapper.F90` calls the actual
unmodified wrapper with moist-theta preparation/finish and all rain channels.
The 12-column, 12-level DT18 fixture includes active cold/hail, warm, missing
number and alternating clear-level regimes. The tracked fixtures and their
hashes are under `tests/v040/milbrandt/`.

MY18: literal noncontracting REAL4 plus scalar libm is byte-exact for all
3168 values, including discarded Dm profiles. MY15: all 279 saved constants
are byte-exact. MY32: changing compiler initialization from zero to signaling
NaN/integer42/logicaltrue leaves all 3168 outputs byte-exact and finite; see
`UPSTREAM_WRF_BUGS.md`. The genuine hail-sedimentation deletion mutant fails
the numerical output assertion.

The optimized comparison retains its original primary NY exceedance:
2772.110107 versus 2772.005859 #/kg, difference 0.104248. Its unchanged
primary tolerance is exceeded. An independently generated original-WRF
36-arm input-ULP envelope bounds it; the candidate is never used to derive
that envelope. The actual wrapper has a separate 38-arm pristine envelope.
MY28 passes the production adapter check after preserving the WRF REAL
preparation stores. WRF discards Dm and does not couple effective radii for
mp=9; the unused optimized Dm_i discrepancy remains disclosed in the docs.

MY29: two real Swiss RD11 production steps (42x42x44, DT18, release FAST=1,
CPU Pallas interpretation, CENSUS=1) completed in 457.9 s including 42.7 s
loading, with zero nonfinite carry values and finite held radar. Nc/Ni/Ns
became active; hail remained inactive in this case and is exercised by the
oracle. Strict guards were enabled: no repair was applied. The single
would-repair qi candidate is -5.337046543178522e-27 kg/kg at step2,
(k,j,i)=(21,22,41), outer boundary ring. It already exists after RK, before
Milbrandt. The WRF spec-zone restriction leaves that ring byte-unchanged.
Interior post-RK minima qi=-1.2158992562794069e-11 and Ni=-13.92313003540039
are absorbed by the original microphysics entry rules. Remaining boundary
Ni minimum is -1.1831570506583722e-14 #/kg.

MY31 matched release mp=8 control used identical inputs, two steps and flags,
with only mp_physics changed: zero nonfinite values, zero guard candidates,
qi identically zero, wall259.0 s/load48.9 s. The qi event is consequently
**not pre-existing in the control**. Its transport/RK birth and unchanged
spec ring establish that it is not a Milbrandt process sign error; it is
recorded as boundary transport round-off under the manager's explicit
standing decision. No guard, repair or tolerance was changed. The coupled
control is regression attribution, not WRF fidelity proof.

MY27 lowers the actual native level-major ABI to CUDA sm120 TTIR on CPU:
0.823 s, 1,100,141 IR characters. It does not execute on GPU. The native
design has one Pallas program per column, fourteen separate field inputs,
and no packed fourteen-field input copy or global discarded-Dm outputs.
Extra preparation round stores may affect fusion/launches; GPU register
pressure, traffic, wall time and peak memory are unmeasured. The earlier
12x12 CPU column receipt reports warm1.119 ms, compile47.763 s and temporary
memory165624 bytes; this is not native-GPU or whole-forecast performance.

MY30 verifies the updated native-facade release probe identity exactly:
`2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`.
Run `proofs/s1_port/release_identity.py TREE OUT.json`, not the superseded
portable-facade identity alone. All experiment receipts are retained under
`<USER_HOME>/wrf_gpu2_lanes/s2-milbrandt/` and logged in main EXPERIMENTS.tsv.

Final composition: MY33 passes 380 checks and exposes four failures: the
public dispatcher omitted mp=9, and three support assertions still pin its
old unsupported status. MY34 fixes those cases and passes all 21 selected
checks, including the actual public mp=9 operational smoke. All 384 MY33
checks are covered after that delta. The three new invalid-selector and
spec-zone round-off controls pass. MY37's clean archive passes the literal
byte and envelope checks. MY38 repeats the updated release identity on the
final composed source: exact hash above, wall60.3 s/load40.3 s/trace19.4 s.
`SOURCE_IDENTITY.json` proves the entire numerical package and five retained
caller/spec-zone function bodies unchanged from the validated model.
`QUALIFICATION.json` retains the small original result receipts. Snapshot,
archive, build trees and caches were removed after their last dependent arm.

Critic fixes (MY39): mp=9 is native-only. CLI preflight and direct daily API
refuse CPU-history replay for omitted, zero and one radar selectors, since
the mandatory MY2 held radar is not retained by replay. Native admission is
preserved. All 209 changed-admission/shared-consumer checks pass (19.16 s),
including nine new cases. This host-only change adds no native kernels,
buffers or transfers; numerical source is byte-identical to reviewed e0b975a9f,
so scientific and default-identity proofs carry forward. The exact manager
F2 instruction is persisted in `MANAGER_F2_DISPOSITION.md` and SHA-bound in
`QUALIFICATION.json`; it explicitly authorizes the recorded boundary-round-off
disposition after comparison, without claiming the event was pre-existing.

Rebased composition: MY40 passes 796 affected checks (54.97 s); after main's
SMS/TKE merge, MY41 passes 345 final restart/admission checks (23.90 s),
including M7 roundtrip and NetCDF locking. MY42's official native-facade probe
retains exact `2bf12cb8` (39.1 s wall). Main's carry and restart files are
byte-identical; catalog/interface entries remain first and mp=9 is appended.
The numerical package and five caller/spec-zone functions remain byte-identical
to the approved version. `REBASE_COMPOSITION.json` records the ordering proof;
`QUALIFICATION.json` binds the tested final composition and receipts.
