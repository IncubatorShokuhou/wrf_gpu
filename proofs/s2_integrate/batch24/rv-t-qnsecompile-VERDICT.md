# MERGE-WITH-FIXES (FIX): QNSE catalog resource disclosure

Reviewed lane/t-qnsecompile at immutable `17943b8b2d3fb56a91fa6c175033900e6fac1fed`.
The compile-fix implementation and numerical gates pass. One required metadata fix remains.

## Required fix

In `src/gpuwrf/io/scheme_catalog.py`, append to the `bl_pbl_physics=4` implemented note the actual c26 GPU E41 result: **FAIL**, PBL STACK **7,232 B** and optional EDMF STACK **2,952 B** per thread versus the 256 B default cap. Identify these as opt-in resource disclosures; keep the default hard gate unchanged. Link the QNSE compile qualification packet. The README/proof packet already disclose the flags, but the catalog note currently omits them. `_WAVE_S1_COMMON.md` explicitly requires opt-in E41 flags in the catalog note. No physics change, tolerance change, or additional GPU/forecast validation is needed for this metadata-only correction; check the changed catalog consumers and carry the exact numerical source proof forward.

## Independently verified

- Production diff is confined to QNSE memory primitives/call sites, with no shared compiler or vendor patch. Five physics/adapter files are byte-identical to the unchanged baseline; the final complete model tree is byte-identical to paid GPU source `4dba72ce4`.
- 10 focused memory/NetCDF controls PASS (7.27 s); replacing actual production typed memory with stock Pallas yields one genuine AssertionError and zero setup errors.
- CPU SM120 native compile SFC/PBL/MF: **0.796/11.046/33.629 s** [M], against the retained baseline's three >90 s Coalesce timeouts. Prescreen STACK 0/9,608/2,584 B and zero calls reproduce the worker's disclosure. All three actual native PTX files contain no f64 instructions, all memory assembly is side-effecting, fills are typed fp32 and KPBL exchanges use int32 operands.
- All **72 fields / 7,380 values**, across 15 staged regimes per driver, are byte-identical in final-source CPU Pallas interpretation. This is regression evidence. Original WRF oracle evidence is carried forward by unchanged equation/adapter bytes; coupled fidelity uses original CPU-WRF.
- Full default identities rerun EXACT: Swiss `f672a573…`; PROD d01 `6fc715a2…`. Initial alias-as-directory launch errors were retained as SETUP_FAIL and counted as no identity evidence; only corrected runs are admitted.
- Retained paid c26 GPU source/receipt [M]: 24 h, 4,800 steps, 25 frames, 519.776 s cold whole wall, 7,054 MiB sampled process peak. Original CPU/GPU actual input hashes match. Verified 20 receipt hashes, four raw evidence hashes, both paid executable pin hashes and 25 original CPU output hashes.
- Independently checked **3,100 scalar gates / 62 bounded fields / 25 frames**, all PASS under the unchanged frozen limits. Full CPU inventory 360 fields; 359 numeric fields finite, 297 report-only, zero missing/schema/nonfinite failures. No forecast repeated.
- Actual paid ELF independently rescanned: 2,360 kernels; QNSE SFC/PBL/MF stacks **0/7,232/2,952 B**, zero QNSE calls. Both JIT and AOT pinned entries contain the same ELF. Raw E41 FAIL is correct; no non-leaf or kInput>500 flags in the retained complete scan.
- Memory semantics retain stock Pallas abstract/discharge effects and indexed pointer calculation. Native accesses are 32-bit aligned; all horizontal accesses use cols<ncol predicates, vertical accesses keep static valid k extents, surface gathers use unchanged clipped k0..9999 plus1 within the 10001-entry table. Output exchange serializes identical replicated-lane writes; discarded old bits are never used. No clamping of masked horizontal indexes. Static transfer scan: zero findings.

## Performance, memory and scientific limits

The source introduces no extra launches, global intermediate arrays or host/device transfers. Typed output exchanges add atomic traffic; actual PBL/MF spilling remains material and disclosed. Cold shared-resource Swiss qualification wall is not a warm S1/PROD benchmark, roofline proof or measured speedup over a working native QNSE baseline. VRAM is sampled, not an exact liveness peak. F2 remains OPEN: 7,854,866 negative phase visits, zero nonfinite visits and 170,344 guard repairs; unchanged guards do not independently attribute every event to original WRF. These global gates are not closed by this compile review.

Review used CPU only, cores16-19 nice19, official RAM6G/no-swap slots and timeouts. Producer source and retained references/pins were not modified. Own disposable caches, native dumps and detached snapshot are removed after sealing. This verdict requests one catalog metadata fix; no review-of-review or duplicate scientific run is requested.
