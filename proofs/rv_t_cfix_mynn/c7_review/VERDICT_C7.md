**MERGE — source/CPU verdict for frozen `353a85f8a11abf4a12dcd78bad77ec7dc682a019`.**

To manager and s2-integrate: no remaining source/CPU blocker. The parallel GPU
gate remains separate under manager 10:17/10:56 instructions; this is not an
unconditional release or performance PASS. Keep the candidate frozen.

Product delta versus 2ed8c04c26 is physics_couplers.py only. The MYNN limited
view uses branch-free selects; level-zero operands are sliced before conversion.
The shared density helper's new kts_only flag defaults False. Its full-column
REAL4 statements and ALT path retain their previous behavior. PSFC staging,
WRF flux limits, shared physical/cumulus handles, and face-w/taper fixes stay
intact. The Exner-enabled pressure substitution is dead input to the unchanged
driver formula; Exner-disabled mode still uses the hydrostatic pressure.

Independent checks on this exact immutable head:

- [M CPU] 13 affected tests PASS, 37 unchanged deselected, 24.74 s: finite
  mixed extreme fluxes, in-range BYTE, branch-free/slice-size structure,
  cu3/cu6/cu16 input preservation, PSFC/previous-EL contract and two-step caller.
- [M CPU] Compiled shared density versus the parent: **8/8 full-column BYTE**
  cases covering fp32/fp64 input/output and normal/ALT outputs. New kts-only
  density is **4/4 BYTE** versus the corresponding full-column level zero.
- [M CPU] Compiled MYNN view versus the parent C5 helper: **18/18 flux arrays
  BYTE, 0 ULP**, covering Exner on/off and inside/mixed/all-beyond limits.
  State HFX/QFX and shared handles remain unchanged. This is CPU regression
  evidence, not an original-WRF oracle or whole-GPU BYTE claim.
- [M CPU] M7–M10 each killed by the intended assertion, zero setup errors:
  added conditional, full-column rho, full-column Exner, and wrong kts faces.
  M10 fails the numerical view gate; M7–M9 fail the required structural gate.
  Snapshot source restored afterward.
- [M source] Coupling source is BYTE to the worker's gated 43c7fd4dda snapshot.
  Physics kernels/formulas and pristine column fixtures remain unchanged;
  their existing independent oracle evidence is reused without duplicate runs.

Independently reproduced official native-facade default jaxpr hashes [M CPU]:

- Swiss: `3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57`
- PROD d01: `196aed9f9973499c5610fdf9965223503a61944156b8cd41357fd8c6e911911e`

Reviewed actual-GPU primary receipts read-only, verified exact source 353a85f8a1,
physical GPU/rc0, complete JIT/AOT stepping coverage, measured rich HLO status,
and native STACK/nonleaf/callgraph results: **Swiss, PROD and WN3-0227 E41 PASS**.
Matched hour-three CUDA traces versus F20 e7e37390cc have equal HtoD/DtoH/DtoD
counts and bytes, synchronization counts and callback counts in all three cases.
This closes the newly added transfer/sync defect; it does not claim zero total
host work. SHA bindings and program coverage are in c7_gpu_audit.json.

At that audit, full 24 h original-CPU frozen D6 receipts PASS for Swiss (25 frames)
and PROD (25 frames per domain, 50 total), complete coverage/no nonfinite values.
WN3 full24 remained pending. These are reviewed v-prep receipts, not rerun here.

Performance/memory: [I source] only kts/kts+1 slices feed the new 2-D algebra;
no new conditional/while, reduction predicate, persistent carry or explicit
host synchronization. The normal shared density path is BYTE. This removes
the former 3-D intermediate/layout link; intermediate liveness and speed are
judged from compiled/runtime evidence, not complexity alone. Actual transfer
counts are verified above. The +5.73% hour-two comparison is cold-compiled
versus pin-restored and cannot decide non-regression. The owner's matched
profiled-hour attribution is supportive, not a substitute for the manager's
fresh-process, cache-loaded PROD ABAB median gate **candidate/F20 <= 1.03**.
No S1, peak-memory, or timing-neutrality claim is made from the source or CPU tests.

Manager/v-prep retain the remaining parallel GPU admission checks on this same
head: matched ABAB timing and WN3 original-CPU D6, plus any terminal gate receipt
they require. PROD numerical acceptance is original-CPU 24 h D6 because its
same-source cold-compile BYTE control was NONBYTE. Do not use GPU-vs-GPU as
scientific truth or treat this source verdict as a waiver of those gates.

Own snapshots/pytest/caches removed; no product edits or GPU runs by this critic.
Earlier sealed decisions are preserved; this C7 verdict is separately sealed.
