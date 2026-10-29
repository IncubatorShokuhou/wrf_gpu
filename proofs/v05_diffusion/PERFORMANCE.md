# D1 performance and memory assessment

All native numbers here are **[M] CPU CompileOnly sm_120 prescreens**, not GPU
execution, throughput, S3 launch counts or end-to-end readiness measurements.
Final model source: `7c83a67469`, based on RC `286e2442a9`. All arms use sealed
PROD inputs, fresh initialization and real force-down before child lowering.

| Program / source | Max stack B | Native kernel bodies | Temp bytes | Argument bytes |
|---|---:|---:|---:|---:|
| d01 all controls, original `75a5` | 288 FAIL | 996 | 466413336 | 126946860 |
| d01 stage-strength fix, no fence `64894` | 288 FAIL | 997 | 466413336 | 126946864 |
| d01 final output fence `7c83` | 32 PASS | 998 | 464619544 | 126946864 |
| d02 final, 3 own steps `7c83` | 32 PASS | 757 | 643027776 | 347000848 |

The final fence changes only the V pressure-gradient output on the unfused UV
path when a vertical-CFL runtime name is selected. It adds one native kernel
body versus the immediately preceding unqualified candidate and reduces peak
compiled temporary storage by 1793792 B (1.71 MiB). The no-fence counterfactual
is source-identical except for the seven-line guard/comment/barrier addition.
There is no field-sized host work or host transfer in the timestep loop.

Host binding adds two REAL stage strengths (8 B logical operands) when dampcoef
is selected. Default REAL values retain the released scalar-fold policy;
nondefault values use a WRF REAL product. Sub-REAL input representations cannot
change those operands or their restart digest. The raw dampcoef operand becomes
dead in the compiled body, giving a measured net argument increase of 4 B versus
the original all-seven-control program. Rebind after finalizing static timestep
and acoustic-step settings; the supported pipeline binds after resolving them.

Cold prescreen compile wall was 196.45 s (d01) and 153.43 s (d02), with whole
setup 480.14 / 462.29 s on shared cores 12,28, nice 19. These observations have no
controlled A/A noise floor. Do not infer a compile or runtime speedup from them.

Actual GPU wall, launches, traffic, synchronization, total VRAM, CPU RSS during
a forecast, and GPU numerical default preservation remain OPEN. The separate
GPU plan requires private OFF/ON caches, real executable E41, explicit NVTX,
changed-value key reuse, paired cost measurements and default byte identity.
The family remains refused by the product. CPU full-root comparison is PASS: 355/355 default leaves are byte-identical,
changed c_s changes 101 leaves with the same two domain keys and two concrete
JIT cache entries. All three arms are finite; peak process RSS is 15234216 KiB
(14.53 GiB), wall 1344.93 s. OFF full-constant/native identity is also PASS for
parent, force-down and child. These CPU results do not certify GPU numerics.
