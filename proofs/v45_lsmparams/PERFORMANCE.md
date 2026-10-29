# W2 resource assessment

Rebased candidate `c1782be2dc` includes main `cb48138349` and the independently
reviewed scoped U-boundary output fence. Its fresh CPU CompileOnly parent
prescreen passes: largest kInput 407, STACK 240 B, no oversized non-leaf
callee, 163 native Pallas calls and zero census errors. Static kernel inventory
is 983; compiled argument/output/temp/alias bytes are unchanged from NATIVE05.
Compile took 127.612 s, whole probe 295.583 s, peak RSS 6,438,012 KiB. These
shared-CPU observations do not measure runtime launches, wall cost, VRAM or
synchronization. The source is byte-exact to reviewed merge tree `7923244a`.
Full-root CPU regression now passes 355/355 default BYTE; all three arms are
finite, CHANGED affects 130 leaves with the same ON domain keys and concrete
cache size 2/2/2. It took 1283.774 s with peak RSS 15,200,332 KiB under the
approved 16 GiB/noSwap cap. This is CPU compile/replay wall, not forecast speed.
Child/actual-GPU native and matched runtime-cost qualification remain open. Receipt:
`NATIVE06_COMPOSED_PARENT/receipt.json` under the lane artifact directory.
The coupled result is `CPU_ROOT03_FIXED/receipt.json` in the same directory.

The following NATIVE05 failure describes the candidate before that fence.

Current numerical candidate `21b4178016` passes the 199-leaf surface and
208-leaf parent-physics BYTE prefix checks, but fails the parent native
prescreen: kInput 574 HLO instructions exceeds 500. Its maximum STACK is
240 B (limit 256 B), with no oversized non-leaf callee and 163 native Pallas
calls / zero census errors. The oversized U-staggered boundary/RK/flux-advection
fusion is assigned to shared-seam lane v45-seam. Child/root qualification is
held. No runtime speed or memory benefit is claimed from this failed prescreen.

NATIVE05 measured 127,954,832 argument bytes, 114,578,436 output bytes,
468,281,152 temporary bytes and 114,576,964 aliased bytes; static kernel
inventory 948. CPU CompileOnly compile took 132.252 s, whole probe 313.914 s,
peak RSS 6,333,948 KiB. These are CPU compilation observations on shared cores,
not GPU execution, runtime launches, VRAM or controlled readiness timings.
All three derived fields total 1,427,004 bytes across the two PROD grids:
four layers of inverse SMCMAX, four layers of dry conductivity and top-layer
inverse DRSURF. They are bound outside the timestep and stripped from the
canopy column argument pack. Host binding/output-clock costs remain unmeasured.

The table below is historical evidence for superseded source `15d898c27d`.
It does not qualify the current source or transfer its native PASS.

Measured CPU CompileOnly prescreens, source `15d898c27d`, captured sm120 target,
autotune 0, real PROD parent 120x70x44, Jarvis2, one own-step program. OFF and ON
use the same source, inputs and options. These numbers are static compiled
resources; they are not runtime launch counts, peak GPU VRAM or speed.

| Resource | OFF | All nine ON | Change |
|---|---:|---:|---:|
| Argument bytes | 126,946,832 | 127,719,632 | +772,800 (+0.61%) |
| Output bytes | 114,578,436 | 114,578,436 | 0 |
| Temporary buffer bytes | 525,258,464 | 472,749,880 | −52,508,584 (−10.00%) |
| Aliased bytes | 114,576,964 | 114,576,964 | 0 |
| Compiled kernel inventory | 1,101 | 1,135 | +34 |
| Maximum native STACK | 32 B | 32 B | unchanged; gate 256 B |

The ON parent and child pass the native census and all native limits. Parent
has 163 native Pallas calls / largest kInput fusion 441 HLO instructions; child
has 153 / largest 330. Neither has an oversized non-leaf callee. The hard HLO
limit remains 500. Actual GPU executables must pass again.

The two derived REAL fields add 317,112 input bytes across the two PROD grids
(2 floats per horizontal cell); this is included in the whole-family argument
count above. They are computed outside the timestep and remain traced clock
operands. The parameter pack is removed before the canopy column arguments.
OFF and legacy maps return unchanged. Output-time clock construction also
uses this seam; its host work/transfers must be included in the GPU assessment.

Observed parent CPU CompileOnly compilation took 147.9 s OFF / 184.9 s ON.
These were separate diagnostic sessions on shared CPU cores; they are not a
controlled compile/readiness speed comparison. GPU compile/readiness,
runtime kernels/copies, host synchronizations, warm cost and peak VRAM/RSS
remain unmeasured for this candidate. The [GPU plan](GPU_PLAN.md) registers the
matched profile and >10% warm-regression stop rule.

Raw receipts and SHA-bound paired summary:
`<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams/NATIVE0{1,2,3}/receipt.json` and
`COMPILED_COST.json`. The default program's complete OFF identity is a separate
E224/E222 gate, not inferred from unchanged arithmetic or this resource table.
