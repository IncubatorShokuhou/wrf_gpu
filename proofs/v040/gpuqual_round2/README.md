# GPU requalification, round 2

This packet reruns the ten configurations in the v-run contract on immutable main
`8afb83d7d`, after the precipitation, microphysics history, CAM-UW and FDDA fixes
and the default Swiss glacier correction. No model code is changed by this lane.

The original CPU-WRF references are reused with matched namelist and input hashes.
Grid FDDA uses t-fdda's corrected `FieldType=104` reference, which actually ingests
the analysis fields. The default Swiss arm uses the original RD11 CPU-WRF case.
Each complete comparison inventories every CPU field in every frame 0 through 24
and applies the unchanged frozen D6 manifest. Missing fields yield INCOMPLETE;
fields without a frozen bound remain REPORT_ONLY. Header differences are audited
separately. Raw pre-mask negative/nonfinite counters are retained, with no blanket
justification of negative visits.

Two independent Swiss GPU processes run concurrently under one official lock;
the parent holds no CUDA context and forwards only the validated lock descriptor.
Every child asserts the GPU backend and the frozen imported source. The actual
cold executable is scanned by the full E41 gate before its disposable cache is
deleted. Minimal exact executable PINs are retained externally for review.

Cold elapsed times and once-per-second sampled per-process VRAM describe these
qualification runs under concurrent GPU/CPU/I/O work. They are not isolated warm
S1 measurements. Opt-in E41 failures are disclosed independently of D6; the release
default retains its hard E41 gate. The tracked helpers preserve the method;
large original CPU/GPU outputs and complete raw comparison reports stay outside
git at the hashed paths in the per-configuration provenance receipts.

The independent original-WRF post-RK observer in t-f2neg (`a7ff1181a`) is
output-byte-neutral across 18,125 field/frame pairs. The manager accepted a narrow
known issue for the preserved Swiss Thompson MP_RE=0 diagnostic and WDM6 magnitude/location classes; selected extrema
and WDM snow/graupel visit-count differences remain OPEN.
`F2_KNOWN_ISSUE.json` binds that source report and scope. It does not provide global
F2 clearance or qualify the other configurations. No targeted GPU capture was requested.

Flag-scope correction: the release enables `GPUWRF_RRTMG_MP_RE=1`. The initial
Thompson arm explicitly used 0 and is preserved as `swiss_re0_diagnostic`, outside
the ten required configurations. The true release-default artifact is
`release_default_re1` (displayed as `release_default`). The completed BL9 and FDDA arms
used 1. The original activation predicate excludes MP16/26/40 for both flag
values; `MP_RE_SCOPE_PROOF.json` records that source-level proof. Their measured
flag values remain explicit; no native-byte equality is claimed.

Storage follows the manager decision at 14:29Z. The finished ROOT snapshot was
removed after its consumers; remaining MP18, MP18+CU5 and MP28 use the identical
Git commit in `<DATA_ROOT>/wrf_gpu2/scratch/v-run/snapshot`, with new caches, temporary
files and GPU output in per-case DATA runtime directories. Completed actual source
paths and artifacts remain unchanged in their historical receipts.

Every arm is bound to the initial frozen main commit `8afb83d7d`. Later composed
main changes require a source-reuse proof or their affected gates; rebasing this
proof branch does not change the source that produced these measurements.
Summary `complete` means all measurements were processed; the separate
`all_required_D6_coverage_pass` flag records the full coverage gate. F2 remains
open outside the explicitly accepted case-scoped known issue.

Final measurements: ten required arms processed, all 62 bounded D6 fields pass
at all 25 hourly frames, and all output fields are finite. Six have complete CPU
field coverage; MP40, MP18, MP18+CU5 and MP28 remain INCOMPLETE on missing fields.
E41 fails independently for MP24/26/40/18/18+CU5/28; release default passes E41.
The final MP28 arm ran alone after resume with a fresh 1500-second timeout.
All disposable caches and both source snapshots have been removed after their
consumers; exact native executables and scientific outputs remain for review.

Metadata parity also remains open: FDDA lacks six global attributes and several
variant scheme attributes retain input defaults. `METADATA_DISCREPANCIES.json`
binds the actual resolved runtime controls and the per-case header discrepancies;
these are audited separately from the registered numerical and field coverage gate.
