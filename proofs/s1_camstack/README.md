# CAM native stack repair

The real Swiss d01 42×42×44 CAM configuration (ra_lw=ra_sw=3, dt 18 s,
radt 10 min, radiation cadence 33, nominal absorption 21600 s) originally had
nine kernels above the 256-byte STACK gate, maximum 2144 bytes. The manager
authorized the original GPU acquisition and a final changed-artifact timing
window. Original target, compiler inputs, native executables and tools remain
in <USER_HOME>/wrf_gpu2_lanes/s1-camstack/ until the critic finishes (E177).

Captured sm_120 target SHA:
`f46c185780fdbd4aab62ef20293385bb8de31a71bd260e9ce62cbfc21f196a23`.
Original precompile MLIR SHA:
`8fd6460e99f900c3862c0fe2efecb59a84aa380cf83264eecc7a679c77bb850f`.
CompileOptions SHA:
`b4be8f13e4381bdae3f1ffd2e41f0e24b762ec156e16002526048fa07dadf1a1`.
Fresh CPU initialization plus virtual-CUDA lowering of the restored original
helpers reproduces this entire 354811247-byte MLIR exactly. Per-leaf committed
sharding matters; imposing sharding on every operand invalidates that control.

`deviceless.py` compiles that input on CPU for the captured GPU target. The
original native artifact reproduces 2144-byte STACK and all nine spill families.
Its static allocation equals the actual GPU allocation 3557441123 B. This is a
native prescreen, not GPU executable byte identity or runtime speed evidence.

The repair preserves WRF expression and addition order: bounded 32-term
interpolation and six-band transmission loops, an integer region scan,
canonical loop-input layouts and completed-result boundaries. Native HLO/SASS
inspection located the mixed-layout producer and residual nearest-layer sum
before adding boundaries. The final V11 CPU native gate passes: STACK 136 B,
no oversized or unresolved nonleaf callees, maximum kInput transpose HLO 404,
and maximum across ALL transpose fusions 479. Static allocation is
1606895467 B versus 3557441123 B originally (-54.83%); this is executable
allocation accounting, not measured peak VRAM. The actual final GPU scan
passes STACK at 96 B and kInput transpose at 479 instructions, with no
oversized or unresolved nonleaf callees. Both stepping entries are covered.
The generic scan retains INCOMPLETE for compact AOT metadata; the separate
completion verifies its exact serialized-blob SHA against full optimized HLO
observed before compaction. No generic verdict is edited or waived.

The old fallback HLO count 397 was wrong: it silently omitted multioutput
headers containing /*index=5*/ comments. The merged scanner exposed a 1846-HLO
producer. Bounding the original serial cumulative sums removed it; a completed
H2O emissivity boundary removed the resulting 288-byte spill. All final native
checks use the merged authoritative scripts/e41 tools. The first paired GPU
measurement exposed another distinction: CompileOnly labeled three bodies
kLoop that actual CUDA compilation labeled kInput (658/1165/513 instructions).
V11 therefore also bounds all transpose bodies on CPU, including the serial
weighted CO2 recurrence and the six independent emission transmission bands.
CPU compilation remains a prescreen; the actual GPU executable must be scanned.

`check_bytes.py` compares every fresh/held CAM output leaf for all 40 active
CAM01 columns against the original CPU port, byte for byte. This is regression
evidence. Independent pristine-WRF LW/absorptivity tests, frozen REAL/r8
tolerances, regime census and genuine deletion mutants are the fidelity authority.
The final V11 regression covers all ten modified helpers. Eight affected LW
oracle tests pass on V11; the absorptivity oracle is carried forward because
its source is unchanged. The Swiss release graph hash remains exactly
2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a.

Tools:

- `capture_gpu.sh` acquires original inputs under the official lock (completed).
- `lower_step.py` initializes the real case on CPU and lowers for virtual CUDA.
- `deviceless.py` retains full optimized HLO/native output and checks STACK/HLO.
- `callee_gate.py` resolves actual CALL-target control flow, distinguishes large
  root callers from nonleaf callees and never passes unresolved flow.
- `measure_gpu.sh SNAPSHOT` acquired the earlier paired ≤900 s GPU arm:
  sequential original/candidate processes, nine warm identical-input coupled
  radiation-step replays and three advancing finite steps. Compilation is
  excluded from replay timing. This is not forecast throughput evidence.
- `measure_gpu_candidate.sh SNAPSHOT` acquires the authorized V11 follow-up
  (≤480 s, start before 2026-10-13 18:45Z), using a private cache and the prior
  original baseline. It never opens the preserved original executable cache.

- `bound_aot_gate.py` supplements compact metadata only with an exact observed
  AOT-blob SHA binding to full optimized HLO retained before compaction.

CPU arms use CPU-only JAX, hidden CUDA, core28, nice19, an explicit timeout,
private artifact paths and the QUIET guard. After the final GPU arm scan rich
JIT and AOT executables with merged scripts/e41 tools before cache cleanup.
Kept resource evidence must satisfy STACK≤256 B, nonleaf callee≤1000 SASS and
kInput transpose fusion≤500 optimized HLO, with complete coverage. The default
Swiss release jaxpr must remain 2bf12cb8. Final evidence seal: STATUS.json.

Replay timing includes a fresh coupled runner and its AOT load on each call;
it excludes input cloning and compilation. It is an API latency measurement,
not GPU-only time or forecast throughput. Allocator high-water measurements
include overlapping master/replay inputs and cannot establish forecast peak
VRAM. The original nine-sample API median is 0.670020336 s. The final
candidate median is 0.683152085 s (+1.96%). Three advancing steps are finite.
Actual cold compile plus first step takes 248.007 s, load 74.234 s, whole
process 330.073 s. Benchmark allocator peak falls 3730029056→2474969088 B
(-33.65%); the 25248645120 B allocator pool is unchanged. Actual static
allocation is 1606880875 B (-54.83%). Static native kernels are 1439 versus
1427 originally; dynamic launch count is unknown. These measurements establish
the resource repair and disclose its costs; they do not establish forecast
speed or peak VRAM. Latest-main composition reproduces the entire compile
input SHA exactly, and its default release graph remains 2bf12cb8.
