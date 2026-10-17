# Upstream WRF 4.7.1 defects relevant to cuplus

## MSKF: exactly calm convecting columns

**Manager decision, 2026-10-13:** the primary MSKF oracle covers realistic
non-calm profiles, including weak but nonzero horizontal wind. Exactly calm
convecting columns are a documented upstream singularity. Port WRF's arithmetic
literally, without a wind floor, clamp, masking or replacement of nonfinite
tendencies. A separate regression must reproduce nonfinite behavior and verify
that the device census/guards report it. No partial READY handoff is authorized.

Pristine `phys/module_cu_mskf.F:5564–5571` computes the mean horizontal wind at
the LCL and midtroposphere, then `TIMEC=DX/VCONV`. At exactly zero wind this is
infinite. The subsequent `NINT(TIMEC/DT)` at line 5617 returned zero in the
measured gfortran production build; `TIMEC` then becomes zero and the returned
tendencies contain NaNs/infinities. The invalid integer-conversion result is
compiler-dependent and is not a physically meaningful finite solution.

The original full `MSKF_CPS` caller was measured with default CMT on, aercu_opt=0,
40 mass levels plus the W top interface, a 5x5 horizontal allocation/3x3 active
tile, dt=54 s, dx=9000 m and WRF REAL constants. Two calm profiles returned
484 NaNs and two infinities in total, despite successful process exits. The same
four profiles with a 2 m/s background wind were finite. Print-only instrumentation
preserved all eight uninstrumented output files byte-for-byte. Raw failures and
the former blocker packet remain preserved; the manager's supported-regime
decision supersedes its policy request, not its measured facts.

Primary numerical comparisons must reject nonfinite candidate and reference
values before calculating errors, and retain their preregistered tolerances.
The calm regression is explicitly separate: expected nonfinite output plus a
positive census/guard count is its pass condition. It must never be included
in a blanket “all columns finite” claim or silently removed from provenance.

Evidence: `MSKF_REFERENCE_BLOCKER.json`; lane artifacts `MS04-outputs`,
`MS05-outputs`, `MS05-full-caller.json`, `MS05-print-only-identity.json`.
The immutable former blocker packet SHA256 is
`16a56132c71f8ae256e5bfbab80c43847a9fc8da094f0cb6a987be9ef0e89fa6`.

### Conversion and reporting regression

`cumulus_mskf_support.wrf_nint_i4` retains the measured gfortran conversion
through INTEGER32 significand/exponent arithmetic. All 4,114 independently
compiled scalar intrinsic samples match, including half ties, finite integer
overflow, NaN and infinities. Standard XLA's positive-infinity cast returns
2147483647 instead of zero; using that cast would silently turn the calm
timescale into a finite value at the original 24-hour cap. No wind floor,
finite replacement, or tendency mask is added.

The exact original timescale block and its port retain Inf/zero at calm wind;
supported nonzero controls include 0.02 and 0.1 m/s. The separate reporting
regression replays actual original-CPS vapor tendencies from both calm
profiles: 40 nonfinite cells each are counted, and the retained finite-state
guard raises without altering their bits. Restoring XLA's saturating cast
causes a genuine numerical assertion failure. Tests and provenance are in
`tests/v040/cuplus/test_mskf_timescale.py` and `mskf_timescale/`.

This component/reporting evidence does **not** qualify the complete native
MSKF CPS or CMT by itself. The later full-column evidence is recorded below;
operational cadence/carries, wiring, restart and coupled gates remain open.

The later **actual full native MSKF CPS with CMT** regression also preserves
the singularity. Against fresh scalar O0/zero-init original WRF, both calm
convecting cases return nonfinite output (260 and 248 reported values in the
expanded full-driver packet). Each native vapor update contains 40 nonfinite
cells: the actual device census counts all 40, reports zero repairs, and the
finite-state guard raises while leaving the array bits unchanged. Two controls
with nonzero winds in their original profiles remain finite. Artifacts:
`MS19-calm-original.json`, `MS20b-calm-native.json` in the lane directory.

## MSKF CMT: additional diagnostic findings

Bounds-checked original builds evaluate `MC(ILEV+1)` in the compound logical
test at line 7847, even on a false branch. Poisoning original local storage also
shows an above-cloud `DER(KQ)` read in the CMT preparation at line 6797: the
no-downdraft branch zeros only levels 1:LTOP. Samples at KQ=6/LTOP=5 and
KQ=7/LTOP=6 retained signaling-NaN bits `7FA00000`. These diagnostics are preserved;
their influence on final physical outputs has not been proved. They do not
justify changing active WRF equations or inventing a finite calm-wind result.

### Compiler initialization policy and sensitivity

The manager's standing policy builds unchanged numerical bodies with
`-finit-real=zero -finit-integer=0 -finit-logical=false`, with FP traps off.
Separate `-finit-real=snan` controls leave FP traps off as well. The port
defines zero at the actual undefined scratch/padding sites and comments their
original locations; this policy does not alter active WRF arithmetic.

The paired complete original-driver outputs are byte identical for eight
CuP columns, eight CAM-ZM columns and twelve MSKF columns. The latter include
eight finite non-calm profiles (0.02 and 2 m/s background wind) and the four
calm controls; two calm controls retain 240 and 246 nonfinite reported values.
Thus these inputs show no material final-output sensitivity to the diagnostic
undefined reads. This finding is limited to the measured columns.

Artifacts: lane `KC14-sensitivity.json`, `ZM04-sensitivity.json`,
`MS09-sensitivity.json`, with separate raw outputs and complete source hashes.

### Literal versus compiled arithmetic

The literal REAL32/scalar-libm executor is a qualification backend for the
candidate equations, not a scientific reference or a production fallback.
The reference remains separately compiled pristine WRF. CuP matches 3,586
defined reported values across eight complete CPS columns. MSKF with its
original DOUBLE CMT island matches all 3,888 reported values across eight
non-calm complete CPS columns against pristine scalar O0/no-FMA arithmetic.
The production vectorized original MSKF build also has a small compiler
spread relative to its scalar build; both original results remain preserved.

CPU Pallas/XLA execution is finite on all eight supported MSKF columns, but
does not pass every field's frozen strict 128-epsilon gate. Measured maximum
absolute deviations against original production O2 are 6.7172e-8 K/s in
RTHCUTEN, 9.1910e-8 m/s² in RUCUTEN, 2.7061e-5 in deep cloud fraction,
1.3784e-7 kg/kg in QI_KF and 1.3114e-6 mm/step in rain. The independently
qualified DOUBLE CMT body itself differs by at most 9.4868e-20 m/s² on the
captured original inputs. These composed compiler deviations are disclosed
under the standing compiler policy; GPU/D6 forecast qualification remains
pending. They are not labelled strict-gate passes or hidden by changing limits.

Artifacts: lane `KC13-literal.json`, `MS14-literal-comparison.json`,
`MS15-cmt-native.json`, `MS17-full-native.json`, `MS18b-full-literal.json`.

## CAM-ZM: original land-fraction convention

`module_cu_camzm_driver.F:459–463` assigns `landfrac=1` when `XLAND==2`,
although the same file documents XLAND 1 as land and 2 as water. The port
retains that literal assignment. It does not silently reverse the source
convention. The independently compiled full caller uses the same input mask.

## Complete callers, real columns and transverse wind

The complete CAM-ZM literal executor matches 5,824 original values across
eight active/suppressed calls with PBL 9 and 2. On CPU XLA all eight are finite;
25 fields pass the fixed REAL32 gate, while three nearly zero RUCUTEN profiles
differ by at most 1.5782e-18 m/s². These are disclosed compiler deviations.

Six 44-level profiles assembled from original Swiss CPU-WRF files add real
temperature, pressure, moisture, geometry and wind structure. The input-bank
manifest distinguishes actual fields from controlled KPBL/ZOL/TPERT and weak
wind variants; these are not captured complete CPS savepoints. Literal full
callers match 15,766 original values. Native CuP six calls and CAM-ZM twelve
calls pass the fixed gate. MSKF passes five of six; one shallow column has two
heat-rate differences up to 1.5170e-9 K/s against a 4.1292e-10 bound.

Separate 35-degree rotations of the active sounding winds exercise both
momentum axes without changing original numerical bodies. Literal MSKF+CMT
matches 1,944 values and CAM-ZM matches 5,832, including full top interfaces.
Compiled outputs are finite but retain small documented field excesses in
the VEC04/VEC06 receipts. None is relabelled a strict numerical pass.
The portable full-caller fixture seals source, driver, compiler and stdout
hashes. GPU and D6 forecast qualification remain pending.
