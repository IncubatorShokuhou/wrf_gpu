# NSSL GPU requalification, round3

Measured source `5abde3224a50e5f67674bb1865aa0cc6d1ff1937`. Original CPU-WRF Swiss 24 h references reused with unchanged input/namelist hashes.

| Config | D6 bounded | CPU fields/frames | Coverage | Raw nonfinite | E41 | Cold wall s | Sampled VRAM MiB |
|---|---:|---:|---|---:|---|---:|---:|
| mp18 | 62/62 PASS | 372/25 | Complete | 0 | FAIL, STACK 1336 B | 920.801 | 2714 |
| mp18_cu5 | 62/62 PASS | 372/25 | Complete | 0 | FAIL, STACK 1336 B | 983.226 | 2720 |

Both cases have zero variable dtype/attribute mismatches and correct physics controls in all frames. TITLE and NTASKS_X/Y differ between CPU and GPU execution provenance.

REFD_MAX equals original CPU-WRF zero in every frame. REFL_10CM is present and finite, but remains REPORT_ONLY: no tolerance is invented. RMSE 0.179069/0.176141 dBZ; maxima 23.737247/19.132549 dBZ.

E41 actual stepping JIT+AOT coverage is complete (two entries/case). Both native artifacts have STACK 1336 B; kInput flags 7/8 (up to 1172 optimized HLO instructions); non-leaf >1k count 0. Exact paid native PINs remain on DATA, with byte hashes in E41_PIN_MAP.json.

F2 remains OPEN: phase-field negative visits 25,775,790/25,841,064 (interior 22,659,657/22,739,467). These are repeated visits, not unique events or roundoff proof. Both raw counter and output nonfinite counts are zero.

Walls are cold concurrent qualification receipts, NOT S1 or isolated performance measurements. Host per-process peak RSS 5.646/5.907 GiB; observed combined scope peak 10.483 GiB <12 GiB, no observed OOM events. Peak VRAM is sampled, not an exact allocator high-water mark.

The f4fd attempt was cancelled before GPU acquisition when reviewed writer-fields merged; its 0-frame/0-compile setup remains externally archived. No physical run was repeated. All four disposable cache/tmp directories were removed after E41. The source snapshot remains only for immediate conditional-program probes.

mp40/mp28 conditional follow-up is pending. This packet qualifies only the measured NSSL cases; E41/F2/global release gates remain open.
