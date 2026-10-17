NTU mp56 / Jensen mp55 native compiler qualification (t-ntucompile)
====================================================================

c87 (Jensen-ISHMAEL, mp55) and c88 (NTU, mp56) never produced a complete GPU
executable: the c88 exact-JIT warm arm hit the 1,770 s cap after one frame and
c87 compiled without an ELF. v-prep's CPU profile put 59 % of the samples in
Triton's TritonGPUCoalesce pass.

Isolated per-kernel CPU compiles (`cubin_probe.py`, CUDA SM120 target, 44 levels,
the production adapter outputs, 4 warps, MLIR pass headers via `pass_filter.py`)
reproduce it for EACH kernel alone: both enter TritonGPUCoalesce within 50 s and
are still there at the 300 s cap (NC01). NTU has only 88 tensor-of-pointer loads
(its 316 table reads, incl. the 98 original-DOUBLE ones, are column-uniform scalars);
Jensen has 664 vector table gathers. Coalesce walks the unrolled column DAG from
every tensor memory operation (FINDINGS E157; MP52/MP53/QNSE precedent).

Fix: `src/gpuwrf/kernels/_column_memory.py` = the qualified MP52 typed-global-memory
technique under its own primitive names (f32 vector loads with masked fills as typed
`ld.global`, f32 vector outputs as race-free `atom.global.exch`, scalars keep the
stock path; stock Pallas discharge rules). `phys_ntu.py` / `phys_jensen.py` swap
`plt.load/plt.store` for it — an exact, reversible substitution. No arithmetic,
level/table indexing, buffers, launch geometry, warps or compiler passes change.

CPU compile (isolated, SM120, no dump, NC04/NC06): NTU 96.6 s compile (ttgir 5.6 s,
LLVM 14.9 s, PTX 12.6 s, ptxas 62.7 s), REG168 STACK4536 B SASS 343,680, two small
leaf callees (138/460 SASS). Jensen 253.8 s (TTGIR 177.9 s, RemoveLayoutConversions dominates:
1,450 category reductions 64x4->64), REG168 STACK5752 B SASS 184,152, no calls.
2 warps for Jensen is worse (403 s, STACK 10,320 B) -> 4 warps kept.

CPU BYTE: the interpreted (CPU) program is identical — lowered StableHLO sha256 of
`full_column(interpret=True)` base vs candidate: NTU 44 lev dc56b5e0 / 12 lev
0bece37a, Jensen 44 lev 485507c6 / 40 lev ceb6fbff (NC03/NC05). Therefore all frozen
pristine-WRF oracle gates and native CPU observations (proofs/s2_ntu,
proofs/s2_mpmore/jensen55) carry over unchanged; their source pins accept only the
reversible substitution. The final f32-only module lowers both kernels to the same
TTIR as the measured candidate (NTU dd1a78aa, Jensen 72f17674, NC06).

Tests: tests/v040/test_column_typed_memory.py (interpretation identical to stock
Pallas; no tensor tt.load/tt.store left in either production TTIR) + evidence pins;
14 PASS; 5 genuine mutants killed (stock loads NTU/Jensen, dropped load mask,
extra kernel edits), 0 setup errors (NC07).

Resources/performance: no new launches, global temporaries or host transfers. The
STACK > 256 B is an E41 opt-in disclosure (wave policy), like P3/QNSE; atomic output
exchanges and local-memory spills are runtime costs measured on the GPU arm.
GPU (one locked session each, cores 3,19,13,29, private caches, original c87/c88 inputs):
- Jensen kernel alone compiled but faulted with CUDA_ERROR_ILLEGAL_ADDRESS (G02; c87 whole
  program likewise after a 4 min 59 s compile). Pallas interpretation gathers with jnp
  semantics (negative indices wrap once, then clamp into [0, dim-1]); raw Triton addresses do
  not, so discarded lanes read outside a table. `_in_bounds` now emits exactly that rule in
  the Triton lowering only (interpretation/CPU unchanged; +0.9 % SASS). Fixed kernel: XLA
  compile 115.5 s, 8.3 ms/call, finite (G03).
- c87 1 h CLI: rc 0, step program compiled 6 min 6 s, whole run 579 s, VRAM 6.55 GB, finite;
  +1 h frame vs original CPU-WRF, frozen D6 manifest 9df4: PASS (385 common variables).
- c88 1 h CLI: rc 0, step program compiled 5 min 11 s (before: >1,770 s), whole run 919 s,
  VRAM 6.0 GB, finite. NTU is SLOW at runtime: 1.85 s/call (+1 h) and 1.35-1.71 s/call (+12 h)
  on real cloudy columns (G04) vs 31 ms clear-sky; its PTX has ~26k fp64 instructions (2,092
  div.f64) from the original DOUBLE islands inside the micro-step loops (1/64-rate fp64).
  A 24 h c88 forecast therefore needs ~2 h of GPU for NTU alone — not a compile problem.
- Actual E41 (wave_cubin_gate, coverage PASS): only STACK flags, jensen_mp55_column 5,472 B
  (REG168) and ntu_mp56_column 3,496 B (REG255); no callee/HLO flags -> opt-in disclosure.
- c88 kernel census (62 unique Pallas kernels, NC11): none stalls any more; largest TTGIR
  b_core_scalar_flux_fp32_stacked 32 s (28 NTU species).
Default programs: Swiss f672a573 / PROD d01 6fc715a2 exact on the rebased candidate.

NTU performance series (b), BYTE-identical outputs
--------------------------------------------------
Manager decision (b): make NTU faster without changing any output bit, so the c88
qualification carries over. Two changes, nothing else:
1. `phys_ntu.py` CompilerParams num_warps 4 -> 2 (one token).
2. `physics/ntu/math.py` find_rc0: a level whose aerosol fraction is NaN skips the bounded
   Newton loop and returns that NaN. Cloud-free kept levels have xafrc = dcn/tcn = 0/0 = NaN
   (WRF calls FIND_RC0 there too); the NaN residual can never converge, so the old loop always
   ran all 50 trips and returned NaN anyway. NumPy census, 16 real c88 +12 h columns: Newton
   trips 10,600 -> 552 (NP02-NP05); the other adaptive loops are unchanged.
BYTE evidence (each link bitwise, all outputs):
- G05 [M GPU] real c88 original CPU-WRF columns +1 h/+12 h (1764x44): 4 warps == 2 warps, 40/40
  outputs per frame. 1 warp also bitwise but slower (discarded).
- G06 [M GPU] same columns: 2 warps == 2 warps + NaN shortcut, 80/80 arrays (`ntu_npz_compare.py`).
- NP06 [M CPU] interpreted production kernel on the frozen original-WRF observation set
  (proofs/s2_ntu/native_probe.py, 12 cases x 2 steps): 912/912 arrays bitwise, same NPZ SHA
  0f70d7e4 (`NAN_SHORTCUT_BYTE.json`). tests/v040/test_ntu_find_rc0_nan.py compares with the
  verbatim pre-change port; the shortcut-removed mutant fails.
- NP08 [M CPU] rebased head: the NTU kernel native TTIR is byte-identical to the GPU-measured
  snapshot (sha 41ef3401); optimized TTIR identical apart from source-path locations.
Speed, s per NTU call (median of warm calls, +1 h / +12 h): merged 4 warps 1.872 / 1.783 ->
2 warps 1.695 / 1.145 -> 2 warps + NaN shortcut 0.389 / 0.286 [M GPU]. Same-session G06 (2 warps 1.688 / 1.142):
4.34x / 3.99x; vs the merged 4-warp baseline (G05, other session) 4.8x / 6.2x.
The c88 24 h arm on the merged 4-warp head took 13,287 s wall for 4,800 steps (v-prep; 25/25
frames, 0 bounded D6 misses, NF 0). NTU was about 1.1-1.9 s of each step; this head should save
about 4,000-7,000 s of that run [I].

3. small_dt skip (`support.masked_call`, emitter `SKIP_UNSELECTED={small_dt}`): NTU spends its time on ONE
   column (small steps median 2/11, max 1064/715 on c88 +1 h/+12 h, NP11), and on those columns 97-99 % of the
   small steps have no active cloudy level (NP12): a cloud-free level keeps stepping in tiny dts. WRF's IF skips
   the microphysics there; the SIMT port computed small_dt for all lanes and discarded it. small_dt now runs under
   lax.cond(any(m9 & alive)); every one of its 29 outputs merges as where(m9 & alive, out, old), so skipping is
   exact by construction. The NumPy literal reference still always calls.
   BYTE: NP14 [M CPU] frozen original-WRF observations 912/912 bitwise, same NPZ SHA 0f70d7e4
   (`SKIP_SMALL_DT_BYTE.json`); G08 [M GPU] real c88 columns 80/80 arrays bitwise vs 5307ed5286.
   tests/v040/test_ntu_masked_small_dt.py kills both mutants (always call / direct call).
   Speed, same session (G08): 0.388 / 0.281 -> 0.223 / 0.224 s per call (-43 % / -20 %); vs the merged 4-warp
   kernel (G05, other session) 8.4x / 8.0x. CPU SM120 compile 54 s, REG168 STACK4672 B (unchanged).
