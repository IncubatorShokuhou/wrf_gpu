# RC N72 graupel tail: measured evidence, 2026-10-29

**Classification: NO, a coupled “trajectory trace, not a port defect” conclusion is
not yet proved.** The tested RC final-hour inputs also produce zero graupel in
pristine WRF. The corrected entry gate is present and consumed in the program
the GPU loaded. Neither fact identifies the exact 71–72 h loss process or proves
the upstream thermodynamic/rain trajectory WRF-faithful. No release gate waiver
or additional model change follows from this packet.

All paths below are relative to `receipts/rc_tail/` unless stated otherwise.
The new analyses ran CPU-only; executable inspection is of the **actual GPU
artifact**, not a CPU or deviceless recompile.

## Program provenance and gate [M]

RC run R4: WN3 0227 (initial valid time 2026-02-28 00Z), N72, 72 h,
2026-10-29 01:34–01:56Z;
source head `254c4823745275e3e9d397ae4d1db0fd35ec361d`, source tree prefix
`827470e1`, incorporating accepted fix `720d8d2937`.
`ACTUAL_LOADED_PTX.json` binds the observed `loaded=true source=aot_blob`
fused_d02 path and execution-log SHA. Executable SHA256:
`2836ab48a7a461abbf568e380ada54a63a266029e660d93000a47523dbf4ae4e`;
native record 2 SHA256:
`a357de08d5137b39577ab8220bc701ce7e9eca3926833df1719bcaaa6de9dba0`.

`COMPILED_GATE_AUDIT.json` records both `thompson_full_column` kernels' complete
PTX chain: entry qg (input slot 5) > R1 AND updated qg > R1, column ANY reduction,
nonzero column predicate, and its use to mask both graupel fall speeds. Child
PTX lines 337/7642/9435/9531/9573/9643/9712 provide the chain; both complete
kernel extracts are retained. The original float32 R1 is unchanged.

Native `NATIVE_GATE_EXCERPTS.json` additionally shows that ptxas retained the
conjunction and REDUX: entry qg register R59 is compared to the literal
`9.9999999600419720025e-13` under the updated-qg predicate P2. Its selected 0/1
feeds REDUX.MAX. Both native kernels have this witness. `ACTUAL_NATIVE_SASS.json`
binds the native ELF and disassembly hashes; the full 9.6 MB temporary disassembly
was reduced to these excerpts and deleted. Inspection used a RAM file descriptor,
not CUDA execution. These are compiled dataflow facts, **not per-cell dynamic
predicate-visit measurements**.

## Last hourly tail [M]

`TAIL_CENSUS.json` contains all positive cells and source-frame SHA256s.
Old GPU counts are regression evidence only; CPU-WRF remains the reference.

| Lead h | RC positive cells | v0.4.4 positive cells | RC maximum kg/kg |
|---|---:|---:|---:|
| 60–65 | 0 | 0 | 0 |
| 66 | 13 | 0 | 2.46496268e-10 |
| 67 | 3 | 72 | 2.35773835e-12 |
| 68 | 0 | 0 | 0 |
| 69 | 236 | 238 | 8.44368797e-10 |
| 70 | 40 | 44 | 1.83633057e-11 |
| 71 | 46 | 7 | 6.40487594e-11 |
| 72 | 0 | 0 | 0 |

All frames finite, with no negative qg. The final nonzero hourly frame has
46 cells in 28 columns, zero-based k=6–11, y=46–50, x=42–53; peak (10,48,44).
`LAST_46_CELLS.tsv` enumerates every cell/value; `COLUMNS_28.json` enumerates
the diagnostic columns. Hourly output brackets disappearance to 71–72 h;
it cannot identify one of the roughly 600 d03 own steps (dt=6 s), distinguish
microphysics from transport/guards, or exclude later sub-hourly reappearance.
The tail changed substantially even though both releases end at zero.

## Pristine WRF on RC final-hour inputs [M]

`FINAL_INPUT_ORACLE.json` / `.npz` hold 21 one-step column experiments:
three locations × seven explicit whole-column input variants, dt=6 s.
`rc_column_inputs.py` is the producer. These use reconstructed hourly states,
**not actual pre-microphysics savepoints**. They diagnose input dependence;
they are not a replay of the final model hour.

The current original CPU t72 file was re-staged through the ALISIOS facade and
proved byte-identical to the earlier G01 reference (16,285,773 B, SHA256
`2152b8b54b1faf15c2c5a64262134fe035c2a2154068dfe5455db77a34b8a254`).
`CPU_REFERENCE_IDENTITY.json` and `ORACLE_BUILD_IDENTITY.json` bind the reference
and pristine driver. The temporary facade copy and 33 MB driver were deleted
after use; original references were untouched.

| Input variant | qg max at column (46,46) | (47,45) | (47,46) |
|---|---:|---:|---:|
| Original CPU | 3.44320e-12 | 1.40823e-12 | 7.68707e-12 |
| CPU with entry qg cleared | 0 | 0 | 1.18217242e-12 |
| Original RC | 0 | 0 | 0 |
| RC with CPU qr,Nr | 1.71577097e-12 | 0 | 0 |
| RC with CPU th,pii,p,qv | 0 | 0 | 0 |
| RC with both groups | 0 | 0 | 1.18217242e-12 |
| RC with CPU entry qg | 2.96676e-12 | 0 | 5.67706515e-12 |

At CPU peak cell (k7,y47,x46), CPU→RC temperature is 267.272→268.406 K,
qr 7.03041e-5→1.42202e-5, Nr 2119.93→525.09. Restoring both rain and
thermodynamic inputs restores the rain-born trace of 1.1821724155e-12 in
pristine WRF. Rain alone and thermodynamics alone do not restore it there.
The differing result in column (46,46) prevents a universal single-variable
explanation. The origin of those input differences remains unproved.

## Preservation, resource consequences, and next evidence

`PRESERVED.json` and `COPY_LIST.tsv`: exactly 13 d03 GPU frames, 60–72 h,
836,482,803 B = **0.779035 GiB**, copied and SHA-verified by 02:35:14Z to
`<USER_HOME>/wrf_gpu2_lanes/v45-graupel/RC_N72_20261029/d03/`.
Minimum ROOT free after copying was 22.357 GiB. No other raw frame was copied.
Completed disposable own caches subsequently freed 954,467,719 B; the old
recoverable source snapshot and driver were also retired. This packet adds
about 1.3 MB of durable source evidence, no production kernel/launch/traffic,
and no new model compile or GPU use. It makes no performance claim.

Needed next: unchanged paid RC executable, selected 28-column observations
during 70–72 h, with exact own-step/process provenance. Fused-call boundary
observations alone cannot resolve the three internal d03 steps or individual
operators. A process observer must preserve and bind the actual native program
and reproduce the retained hourly qg values; E242/E254 forbid inferring this
from a CPU smoke. Until such observations and same-input pristine comparisons
exist, coupled trajectory classification remains OPEN.
