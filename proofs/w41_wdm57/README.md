# WDM5/7 Nn column-view firewall

WDM5 (mp14, c10) and WDM7 (mp26, c18) have the same column-major CCN seam as
the reviewed WDM6 c11 compiler cliff. The product change wraps only their
`mp(state.Nn)` arguments in the existing `small_grid_firewall`. It changes no
microphysics statements or State fields. Grids above 4096 columns trace no
additional operation. The entire adapter module's AST matches the baseline
after removing those two identity wrappers.

This is a **compiler-enabler packet**, with completed CPU checks and two actual
GPU200-step finite-value smoke tests. It is not WRF forecast qualification.
Independent critic returned MERGE/no required fixes on24beaf75c7; the final
c10 smoke is an evidence-only supplement with unchanged model/test bytes.

## Measurement method

Actual sealed v-prep Swiss 42×42×44 composites, one history segment (200 steps),
radiation cadence 33 steps; c10 mp14/cu95/bl16/sf1, c18 mp26/cu0/bl5/sf5,
both Noah-MP and RRTMG4/4. The c11 compiler harness loads the real inputs on CPU,
restores native tracing, and compiles the full program with the captured RTX
5090 target through CompileOnly. AUTOTUNE0; FAST1/CENSUS1/SLIM0/AOTSLIM0/MP_RE1;
CPU24,25,28,29 nice19, 6 GiB RAM scope, timeout900, assembler limit300.
Every receipt asserts physical CPU devices; CUDA_VISIBLE_DEVICES is empty.

Baseline: immutable `47c50d376b`. c10 candidate: `a06135cdc7`; c18 candidate:
`ef39001412`. Their model source is byte-identical. Later commits add only tests,
proof tooling and notes. Default identity uses the separate official
FAST1/CENSUS0 recipe.

Assembler wall spans and RSS are **sampled once per second**. These are shared
CPU measurements, not isolated compiler benchmarks or GPU speed measurements.
The unfenced baselines already assemble in about13 and22 seconds: neither
reproduces WDM6's severe assembler stall. This is a targeted HLO/stack compiler
enabler with lower sampled assembler RSS, not a demonstrated compile-time
speedup. Runtime benefit remains unmeasured.

| [M CPU] | c10 before → after | c18 before → after |
|---|---|---|
| Full module compile, seconds | 224.357 → 222.363 | 273.597 → 373.466 |
| Sampled assembler span, seconds | 13.242 → 13.279 | 22.067 → 24.247 |
| Sampled assembler peak RSS, KiB | 710216 → 620800 | 723132 → 699284 |
| Compiled temp allocation, bytes | 223180448 → identical | 224576528 → identical |
| kInput transpose bodies over500 HLO instructions | 875 → none | 1211 + 618 → 625 |
| HLO fusion count | 3843 → 3839 | 3940 → 3941 |
| Native kernel definitions | 2587 → 2589 | 2676 → 2677 |
| Native STACK over256 bytes | 280/536/400 → 536/400 | 904/360 → 368 |

All argument, output and alias byte totals also stay exact. c10's remaining
536/400-byte flags have identical canonical HLO bodies and measured stack sizes
before/after. The overall c10 E41 prescreen remains FAIL on these existing
flags. c18's secondary flag persists with618→625 HLO instructions and360→368
stack bytes; its body is not claimed unchanged. Neither option is an overall
E41 PASS. No non-leaf callees exceed1000 SASS instructions in any of the four
compiled programs. c18's compile-wall increase under shared load is measured
and unassigned; it is not hidden by the lower assembler RSS.

c10 canonical fused computations: 3076 common of3133 old/3131 new, with57 old
and55 new bodies. c18:3178 common of3181 old/3182 new;3 old→4 new bodies,
including the1211-instruction transpose body→1194-instruction regular body
and an isolated10-instruction transpose. The firewall can repartition floating-point operations on GPU;
CPU byte equality is not a GPU-byte or original-WRF fidelity proof.

## CPU checks

- Initial64 PASS include60 unaffected structural/WDM6/WSM-caller checks and
  four wet comparisons whose constructed theta could default to float64. Those
  four are limited regression evidence and do not qualify nativeREAL32 carry.
  A further4 radius-enabled cases on that earlier fixture also passed.
  [component_carry.json](evidence/component_carry.json) records the60 carried
  passes and exact structural-test AST/shared-test source proof.
- Corrected fixture explicitly types theta float32 and asserts theta/qv dtypes
  on entry and exit through the registered operational dispatch. WDM5/7 already
  use a documented fp64 precision island inside this REAL32 State boundary;
  these checks do not claim native REAL32 scheme arithmetic. A raw-adapter
  attempt bypassed that island: WDM5 four passed, WDM7 four failed during
  tracing with mixed carry types. That fixture attempt is retained separately.
  Eight replacement wet comparisons cover nonuniform CCN, warm/cold levels, both
  precipitation return modes and radii off/on. Every returned array must be
  finite and byte-identical. All8 PASS in111.28s. Deletion control produces2
  genuine assertion kills and0 setup errors. The component count is68:
  60 carried+8 fresh, excluding the earlier mixed-theta wet comparisons. See
  [component_final.json](evidence/component_final.json). The full shared-consumer
  FAST0 run passed336/336 with no errors or skips in1834.41s; see
  [cpu_shared.json](evidence/cpu_shared.json). Both official frozen defaults match EXACT:
  Swiss3356e249…7c57 and PROD196aed9f…911e; full hashes and receipts are in
  [default_identity.json](evidence/default_identity.json).
- The initial full shared-consumer process hit its6G hard cap after23 progress
  dots and produced no JUnit report. Those dots are not counted as passes.
  `evidence/consumer_oom.json` retains the systemd `oom-kill`/6GiB peak proof.
  The12G recovery completed both identities. Its shared suite accidentally
  inherited FAST1 from the identity recipe, overriding `tests/conftest.py`'s
  CPU FAST0 default:332 PASS,4 FAIL,0 ERROR in1761.82s, with10GiB sampled
  process high-water RSS. One KF fixture lacks held rates and three restart
  fixtures lack held effective radii under FAST1. This remains an extra FAST1
  compatibility result, not a passing required CPU gate; see
  [shared_fast1.json](evidence/shared_fast1.json). The full unchanged336-test
  selection passed under explicit FAST0/CENSUS0/x64 with the established
  per-test executable-cache cleanup and12G cap. All336 test setups had x64
  enabled; process high-water RSS was10,902,244KiB. No model code or assertions
  were changed to accommodate the earlier failures.
- A live-driver edit raced queue admission and caused shell EOF after the older
  four radius cases passed. This setup failure is retained in the ledger. The
  replacement driver and helpers run only from immutable snapshots.

## Reproduction and retained evidence

Completed compiler packet: [compiler_evidence.json](evidence/compiler_evidence.json),
SHA256 `a37a5b92db1a9de797d3851cb6e9321df970348ac9d791cb728e682c1b0dec79`.
It includes exact input/work hashes, compiled memory totals, native ELF hashes
and full remaining E41 flags. It is independent of the completed CPU gates.

`probe.sh before|after c10|c18` runs the unchanged actual-composite compiler
recipe; use detached snapshots at the commits above in the lane's `baseline`
and `candidate` directories, with `data/wrf_pristine` linked in both. Every run
must use the QUIET check and `with_ram_slot.sh --max 6G`, with its own timeout.
`summarize.py ARM` saves compressed HLO/buffer reports and scans the serialized
executable; `compare.py CASE` compares identical inputs/work and canonical fused
bodies. `cpu_finish.sh` is the retained original6G attempt: its component and
mutant stages passed, then its combined shared suite hit the recorded cap.
`cpu_resume.sh` retains the FAST1 compatibility attempt. Use
`cpu_shared.sh PROOF_COMMIT` from an immutable proof git blob in an official
12G CPU RAM slot for the required FAST0 full shared consumers, with per-test
executable-cache cleanup and x64 baseline restoration. Completed components,
mutants and identities are carried forward. The measured model stays at immutable
`final-candidate` snapshot4474617f25 throughout. That snapshot was removed
after its last dependent arm; recreate it at4474617f25 to reproduce these runs.
`source_proof.py` verifies the two identity-wrapper edits against the baseline.

`gpu_one.sh PROOF_COMMIT MODE` runs one official lock session from an immutable
proof git blob, with no RAM slot around or inside the GPU lock. Its GPU child
uses a20G/no-swap systemd scope and an explicit timeout. Existing CENSUS1
pre-mask/actual-work counters are read once after a completed200-step segment.
`gpu_post.sh CASE` runs the CPU cubin scan in a normal RAM slot after unlock;
postprocessing can wait while the next GPU arm runs. Private compiler caches
are cleaned after their last dependent arm; native binaries survive until scan.

## October17 GPU evidence

GPU `compile_s` includes lowering plus backend compilation. The CPU table
times module compilation after lowering; the two timings are not a paired
CPU/GPU compiler speed comparison.

The first actual c10 GPU lower/compile took312.735s. Its execution was refused
before any forecast step because the probe omitted production's
`unalias_donated_carry(..., by_buffer=True)` admission step; this is a probe
error, not evidence against the firewall. Fixed probe commit91aff6f75b adds
that normal admission and a unique-buffer check. The72,149,482-byte native
binary was preserved with SHA256
`84e0ae8489786261ef35e23f912340231ca4903bd726680d39ba440d68b46a9d`.
Its full real-binary scan has1/1 coverage, no transpose body over500 HLO
instructions, no non-leaf callee over1000 SASS and no unknown callgraph.
The remaining536/400-byte stacks keep overall E41 FAIL. See
[gpu_c10_cubin.json](evidence/gpu_c10_cubin.json).

The corrected replay loaded that exact native binary with source/input/ABI/
allocation guards and no root backend compile. Initialization plus lower/load
took230.851s, leaving about9s of its240s cap for execution; it entered execution
and timed out before completing the segment. F2 and segment runtime remain
unknown for that attempt. The final300s replay completed using its private
initialization cache: initialization/admission29.494s, lower/native-load148.215s,
execution entry177.768s, segment16.623s, final process receipt194.432s and
lock197s. The exact scanned binary ran200 steps/1400 acoustic trips/7 radiation
tendency calls plus1 init call, with all201 returned leaves finite and observed
pre-mask nonfinite0. Negative phase visits335,483 include271,667 interior visits;
F2 remains OPEN. No root backend compile or duplicate cubin scan was needed.
See [gpu_c10_smoke.json](evidence/gpu_c10_smoke.json).

The separate c18 cold arm completed: lower/compile299.719s, segment15.032s,
200 steps,1400 acoustic trips,7 radiation tendency calls plus1 init call;
all188 returned leaves finite and observed pre-mask nonfinite0. Its589,874 negative
phase visits include475,669 interior visits, all after RK, so F2 remains OPEN.
The census covers15 fields at4 phases and omits WDM7's active hail `qh`;
zero observed pre-mask nonfinite events therefore does not qualify hail
pre-mask values. This coverage gap is a separate open F2 issue.
The c18 actual native scan has1/1 coverage and confirms the residual
625-instruction transpose/368-byte stack, with no non-leaf callee over1000 SASS
and no unknown callgraph; overall E41 remains FAIL. See
[gpu_c18_cubin.json](evidence/gpu_c18_cubin.json) and
[gpu_c18_smoke.json](evidence/gpu_c18_smoke.json). These are component segment
measurements, with no GPU baseline pair or original-WRF comparison.
Total official GPU lock time across all four attempts was1263s (21.05min),
within the final25min grant. The final replay completed, so the conditional
October18 smoke retry is unnecessary. Both opt-in E41 gates and the scientific
F1/F2 gates remain open; neither segment has an original-WRF comparison here.

Working artifacts: `<USER_HOME>/wrf_gpu2_lanes/w41-wdm57/`. Raw compile caches,
PTX and serialized executables are deleted after their last dependent scan/replay;
compressed HLO/buffer reports, receipts and scan JSON remain. Final native
launch/traffic counts, VRAM, runtime and affected
original-WRF forecast validation remain for the manager's composed GPU window.
Static fusion counts and native kernel-definition counts are not dynamic
launch counts. No runtime speed or peak-VRAM claim follows from these data.
All disposable CPU/GPU caches, native replay copies and measurement snapshots
were removed after their last dependent arm. No inputs or references were removed.
