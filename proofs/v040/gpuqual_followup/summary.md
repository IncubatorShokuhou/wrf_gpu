# Conditional qualification follow-up

MP40's full selected native step program, all 61 constants and 175 ordered initial array inputs match measured source 8afb exactly. Each source matches all 134 checked scalar controls from the original GPU receipt. The Python wrapper differs only by added None slots. This proves model-program identity; it does not attest new CUDA executable bytes.

Retained MP40 files were augmented using the admitted writer provider for 11 WRF-inactive zero fields and the current metadata table. All 9,500 original variable/frame pairs remain byte-identical; all 275 new constant/frame pairs match original CPU-WRF exactly. The 62 bounded D6 results are carried forward without repeating numerical validation. Coverage remains INCOMPLETE only for MSKF_REFL_10CM, which is uncomputed and was not filled. The host-only operation took 11.692 s including copying and strict byte/schema checks, with peak RSS 306,100 KiB. This is not an operational writer benchmark.

MP28's selected model program changed (017556ca to 220063e1), with its actual initial array inputs growing from 180 to 182. All 134 checked measured controls still match. The manager approved one full 24 h GPU rerun, capped at 20 minutes. It is queued on the next-marker behind v-prep, using immutable main 195cd4194. Its model/data/scripts trees match the captured 5ab program exactly.

No original reference or science file changed. Completed 8afb and 5ab snapshots and all identity/NSSL disposable caches were removed. Only the pending MP28 snapshot remains until its last consumer. E41 and F2 retain their case-specific open findings; no release clearance is claimed.
