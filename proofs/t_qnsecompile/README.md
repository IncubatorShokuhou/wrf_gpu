QNSE compiler qualification
===========================

All three 44-level QNSE column kernels stalled in TritonGPUCoalesce
individually for more than 90 seconds. The previous complete c26 forecast
attempt timed out after 1,770 seconds during compilation.

The fix uses QNSE-only typed global memory primitives, following the
qualified MP52 technique. Pallas reference effects and CPU discharge rules
stay unchanged. REAL4 input loads retain masked fills; output exchanges
preserve REAL4 and integer KPBL bits and avoid races from replicated lanes.
No compiler pass, vendor library, physics equation or kernel boundary changes.

The CPU native probes compile surface/PBL/mass flux in 0.94/12.97/38.99 s.
CPU interpretation preserves all 72 fields / 7,380 values across 15 staged
WRF regimes per driver. These are regression checks; the original pristine
WRF oracle evidence is carried forward by exact physics/adapter source bytes.
Both full default program identities remain exact. Ten focused controls pass;
removing typed memory from production QNSE triggers a genuine assertion.

Native PBL/MF stacks are 9,608/2,584 bytes per thread, exceeding the default
E41 cap. This remains an explicit opt-in resource disclosure, not an E41 PASS.
Output atomic exchanges and spill traffic need actual GPU timing. The change
adds no source-level launches, global intermediates or host transfers.

`CPU_QUALIFICATION.json` seals the completed CPU gate. The single authorized
c26 Swiss 24-hour GPU arm completed in 519.78 s cold, writing all 25 frames
with 7,054 MiB sampled process peak VRAM. `score.py` uses the sealed original
CPU-WRF outputs and unchanged frozen D6 scorer/bounds. `audit_cache.py` scans
all paid JIT/AOT programs and retains exact QNSE ELF/native resources before
disposable-cache cleanup. All 62 bounded D6 fields pass over 25 frames;
297 numeric fields are report-only. Coverage is complete, with no missing
CPU fields or nonfinite values. Actual QNSE surface/PBL/MF stacks are
0/7,232/2,952 bytes, so the full E41 result remains an explicit opt-in FAIL.
Both paid JIT and AOT entries contain the identical native ELF and are pinned.
The census records 7,854,866 negative phase visits and zero nonfinite visits;
F2 source attribution remains OPEN. This packet closes the compile lane,
not global F2, E41, performance, memory or release gates.

`QUALIFICATION.json` is the final receipt index. Disposable CPU/GPU caches,
prepared scratch inputs and detached snapshots are removed. Original CPU
references, paid executable pins, the actual native ELF and forecast/D6 raw
evidence remain available at the recorded paths for independent review.
