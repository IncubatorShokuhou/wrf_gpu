# Composite CUDA preflight on CPU

`sweep.py` visits every sealed v-prep matrix row, prioritizing rows without an
actual GPU receipt. Each `lower_case.py` process runs on 8,9,24,25 with nice 19,
a 900-second timeout, MemoryMax=6G and MemorySwapMax=0. The supervisor checks
QUIET before every launch and cleanup and appends experiment/failure messages.
Heavy children now acquire the manager's global `with_ram_slot.sh` lock first,
then recheck QUIET inside the slot. This bounds the nightly aggregate to two
6-GiB jobs. `--reuse-from OLD --rerun c65 c66 --first c65` resumes into a new
output directory, retaining completed receipts and the reboot-interrupted
attempts without overwriting them. The interrupted c65 row is not a real timeout.

The measured source must be a clean detached worktree with the pristine-WRF
symlink. Loader initialization uses CPU-native arrays and the portable Pallas
facade; tracing holds the shared platform capability resolver at CUDA (including
directly imported selector aliases), restores native Pallas, GPU selectors and the
original product tridiagonal primitive. A captured CUDA target supplies the
CompileOnly client. No physical GPU client is created or used.

Single-domain programs use the actual CLI history-segment chunk size and bound
radiation cadence. Nested children first receive the product force-down output;
per-domain programs and any eligible product fused cascade are lowered. The
namelist, clock and start index remain dynamic arguments, matching the runtime
signature. This does not execute model steps, feedback, output or DFI trajectories.

Trace-time backend changes clear JIT caches. The c48 control compares native IR
before/after that clear; the actual nested jaxpr is checked recursively for any
Pallas call left in interpret mode. DFI additionally captures and lowers its first
real stage arguments, stopping before any model advance; later filtered stages
remain outside this receipt. Slow children emit in-process stacks after 120 s.

Programs under 4 MB StableHLO, plus dry c01 and simple-forcing c39, attempt JAX's own CompileOnly native
compile with autotuning disabled. This is an explicitly different optimization
setup from a production GPU compilation; estimated E41/STACK results cannot
replace the actual GPU gate. Larger programs retain UNKNOWN compile/STACK
results, preserving GPU time by catching frontend failures first. No runtime
speed, memory scaling or scientific fidelity is established by this tool.

`RESULTS.json` and `RESULTS.md` retain each row's source/input/target/tool hashes,
lowering result, native call-site count, process RSS and optional compilation.
Refusal controls are labeled as controls, without asserting their exception is
the intended refusal. An interrupted row is never silently overwritten.

The successor packet `receipts/c65_completed/QUALIFICATION.json` closes the
deferred c65 row: all 76 admitted cases lower successfully. The original packet
remains sealed. c65 records unoptimized constant sizes and a compact IR hash;
large literals are elided, so that hash does not identify the complete HLO.
`sweep.py --ram-max 12G` is reserved for an explicitly authorized singleton.
