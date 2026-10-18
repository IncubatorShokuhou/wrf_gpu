# wrf_gpu2 → alisios_mgr, 2026-10-18: v0.4.1 tuple (DRAFT for the manager — numbers filled from receipts)

| Field | Value |
|---|---|
| commit | `104c97de98b45b790058d3cdad6e0ff1e88a9f82` (v0.4.1 source freeze; the tag points here or at a docs-only commit with the same src tree) |
| src tree | `319848e91a5ecaf5516d0a783bad11447c349049` |
| gpuwrf tree | `216ebe594643ffa845183d6e9a818f3b54cd8323` |
| bundle_sha256 | `24a7b24f2373123a8fd8e68d7d93efc66168d7cb163b2077eb66b88c940ead37` |
| manifest | `45ac664272263af82a1b00654ceebe3961b485e19e7cacb78a4ab98429229c31` |
| tools | P/tools/compile_bundle.py `c704d3d9…` (same as v0.4.0), bundle_consumer.sh `816f5f5b…` / .py `48cfbf0c…` (same as the v0.4.0 consumer), pinned in P/tools/SHA256SUMS |

**Paths, use them verbatim:** P=`<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1` · cache P/jax · source P/src · env P/bundle.env ·
usage P/meta/BUNDLE_USAGE.md · tarball P/sealed/gpuwrf-bundle.tar.gz (2.2 GB) · consumer P/tools/bundle_consumer.sh.

**What changes for you vs v0.4.0:**
- One bundle now serves every issue date of the WN3 3-nest grid (the d02/d03 program no longer bakes Noah-MP TMN).
- 48 h and 72 h input files of one geometry hit the same programs (record-count independent step programs).
- The consumer wrapper ships inside P (P/tools), identical to the v0.4.0 one you use.

**Primed** [M, GPU, official lock, release defaults]: Swiss d01 3 h, PROD 2-domain 3 h, WN3 3-nest 0227 72 h
(`--domains-from-namelist --emit-initial-history`; your G2 C0 namelist equals this namelist except the dates).

**Replay from the sealed bundle (consumer, fresh layer each)** [M]:
| run (fresh process, consumer, fresh layer) | result |
|---|---|
| WN3 0227 72 h (priming date) | all 219 wrfout byte-identical to the cold priming run; d01 + fused d02/d03 loaded from the bundle; 0 compiles |
| WN3 0502 3 h (2nd date) | same d01 and fused programs loaded; 0 compiles |
| WN3 0614 48 h (3rd date, your G2 length) | same programs loaded; 0 compiles |
| WN3 0227 3 h on 48 h input files (8/9 records) | same programs loaded; 0 compiles; frames 0-3 byte-identical to the 72 h-input run |
| Swiss d01 3 h / PROD 3 h | byte-identical to their cold runs (4/4, 6/6); 0 compiles |

Every arm: `compile_bundle check` BUNDLE_OK missing=0 changed=0 new=0 before and after; `BUNDLE_CONSUMER ... compiled_new=0`.
CPU dry run `gpuwrf aot-key`: 0227 vs 0502 / 0614 / 48 h-input share all programs (d01, d02, d03, fused/d02).

**Disclosure:** non-primed configurations (other physics/nudging/domains) still compile once; the bundle removes
codegen variance only — trajectories need D6 against the original CPU-WRF (WN3 0227 72 h from this bundle vs the original CPU-WRF 72 h: d01/d02 PASS; d03 RAINNC leads 69-72 exceed 1 mm (1.27-1.73) but stay below CPU-WRF's own 1-ulp twin spread (1.51-2.11) -> floor-limited, same class as v0.3.3/v0.4.0 and smaller; output integrity PASS).
