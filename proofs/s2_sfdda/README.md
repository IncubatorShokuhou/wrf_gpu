# s2-sfdda qualification

Scope approved by the manager: surface modes 1/2 with PSU grid FDDA=1,
MYNN PBL/SFCLAY=5 and Noah-MP=4. Classic Noah FASDAS soil coupling remains
refused. No GPU qualification claim.

`oracle/build.sh BUILD` compiles a lane-local, unmodified pristine WRF
`module_fdda_psufddagd.F` and original `module_model_constants.F`, both
REAL4 and REAL8. The full `FDDAGD` call includes its own `SFDDAGD` call;
the wrapper preserves step-1 overwrite, STEPFG/fgdtzero, input alarms,
mass coupling, specified ownership, and final-mass decoupling from the
existing o2-fdda caller oracle. Surface OLD is never overwritten on step 1
(that code is commented out in pristine `module_fddagd_driver.F:275`).

`oracle/make_savepoints.py BUILD OUT` uses original CPU-WRF Swiss hourly
columns (5×6×44). Deliberate, labelled perturbations activate stable/free
convection, roughness branches, shallow/deep PBL, observation-density
smoothing, and supersaturated targets. Modes 1/2 each cover plain, positive
and negative ramp, held/zeroed tendencies, and level weights in REAL4/REAL8:
20 base cases plus 4 wind-census cases × 9 caller invocations. The additional
cases use two-cell regime/roughness blocks and a first half-layer >10 m to
exercise both U/V convection corrections and both roughness branches.
These are operator fidelity fixtures,
not a full forecast reference. Source hashes and nonzero output census
are in `savepoints/MANIFEST.json`.

`tests/v040/sfdda/` checks independent oracle rates and the coupled/owned/
decoupled stages, real NetCDF input validation, and public restart formats.
`smoke.py OUT MODE` creates private Swiss input symlinks plus a `wrfsfdda`
file from original CPU-WRF fields, loads through the real CLI loader, runs
two production CPU steps, and checks byte-exact restart continuation.
All CPU work requires JAX_PLATFORMS=cpu, GPUWRF_JAX_CACHE=0, lane CPU affinity,
QUIET guard, timeout and a detached source snapshot.

The literal REAL4 expression control uses an unmodified pristine build with
`SF_ORACLE_FLAGS='-fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off'`
and scalar EXPF in `literal_real4.py`. Its four first-call configurations
(plain and both-face wind, modes 1/2) are byte-exact. Production keeps JAX's
exponential and is checked against the original optimized WRF fixtures.
`saturation_operands.json` and `saturation_linkage.json` localize the small
compiler spread to `_ZGVbN4v_expf`; constants and pre-EXP arithmetic are BYTE.
Neither tolerances nor production math are changed to conceal that spread.

Default release identity must remain Swiss `2bf12cb8`, checked using
`proofs/s1_port/release_identity.py`. GPU timing, E41 and full D6 nudged
hindcasts remain pending.

The final surface carry inherits the grid FDDA history carry. `HFX_FDDA` uses
the final total temperature rate and remains held across skipped, zeroing and
early-return calls. `oracle/make_hfx_savepoints.py BUILD OUT` replays the 24
frozen fixture operands through pristine FDDAGD with caller `dz8w`, producing
216 independent history fields; every one of the 5,616 prior output arrays
stays byte-identical. HFX fixtures are tracked and checked in a clean archive.

The final coupled source is `ca9e3a78e`, a minimal successor to reviewed
`546623868`. Its only model change is host input validation in `wrfsfdda.py`:
all ten OLD/NEW fields must remain finite after REAL32 conversion, and a
shared radius helper follows WRF's domain-wide mode-1 inheritance rule.
SF35 passed 99 targeted host, FDDA, contract and legacy tests in 7.51 s,
including 20 signed DOUBLE overflow probes and three mixed-mode menus with
all-mode-2 negative controls. The independently checked 374-test baseline
and unchanged operator/oracle evidence remain valid.

SF36/SF37 reran the complete production CPU step on this final code for
surface modes 2/1 against the preserved, independently run original CPU-WRF
references. Each completed two steps with finite full carry, all 66 guard
counters (22 × 3) zero at both steps, and five frozen bounded fields passing:
T/U/V/W/QVAPOR. P and PH are finite report-only measurements. Each compiled
restart continuation matches all 208 leaves byte for byte, with identical
carry/resumed SHA256 hashes and JIT execution recorded. The cold CPU arms
took 288.09/331.28 s including compile/restart validation; these are not warm
forecast or GPU speed measurements. Combined process exit was zero; peak
host RSS was 12,954,148 KiB for both sequential arms and compiled programs.

`coupled_mode1_final.json` and `coupled_mode2_final.json` contain the final
coupled evidence. The older SF22 mode1/mode2/control receipts are retained
as prior-source evidence only, with corrected 66-counter/five-bound scope.
Every original numerical datum is preserved (`RECEIPT_CORRECTION.json`).
AST equality is not used as a substitute for the final compiled coupled gate.
`CRITIC_FIXES.json` maps all four rejection items to their closure evidence.

Original CPU mode1 wrote all three reference frames and SUCCESS, but its
cold-table launcher expired and its orphan MPI finalization errored. Its
numerical outputs are retained without a clean-process claim. Original
CPU mode2 and control exited cleanly. These short coupled runs establish
caller operation, not 24-hour drift or GPU qualification.

Final CPU-only 5×6×44, 20-sample component medians: grid-only 0.0412 ms;
surface1 0.0774 ms; surface2 0.0761 ms. Compile costs 0.131/0.291/0.298 s,
temporary bytes 896/6456/11912. Surface adds 1532 carry bytes and 2576 bytes
for two resident analysis records; HFX storage is shared with the grid-only
baseline. These measure the changed artifact on CPU, not GPU launch count,
VRAM or forecast speed. Default native-facade full SHA remains exactly
`2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`.

The default graph identity is reused from the independently verified critic
RV03 full `2bf12cb8` hash. The eight protected source modules and both
surface-off early-return guards are byte-identical to the reviewed tree;
the changed host paths are unreachable for the default surface-off menu.
No unchanged forecast, oracle, component-cost or default-trace run was repeated.
