# w43-conv: amendment 1 §3 contingency tables (light rain, LWP) — rain FINDING (one case, near-threshold), LWP NOT_A_BUG

Manager task 2026-10-21T22:14Z. Rules frozen in `<DATA_ROOT>/alisios/state/manager/cross/FROM_WRF_GPU2_V043_AMEND1_ACK_20261021.md`
§3; none changed here. CPU only (cores 24,25,28, nice 19, MemoryMax 6G), no GPU, no product edit. Script
`analysis/contingency/contingency.py`, full numbers (per case × domain, rate classes, input SHA-256) in `analysis/CONTINGENCY.json`.

## Method (as frozen)
- Inputs: teacher extracts, 8 cases. WN3 0120/0220/0227/0502/0608/0614: `w43-teacher/te_gpu/<issue>_gpu` vs
  `te_repro/<issue>_cpu`. ERA5 0218/0302: `te_gpu` vs `te_cpu`. Domains d01_crop (15×17), d02_crop (35×41), d03 (93×111); τ1–72.
- Events: (a) `rain_rate_mm_h` ≥ 0.1; (b) `LWP` ≥ 0.01 kg/m². **LWP is a proxy**: the extract's LWP is the column sum of
  (QCLOUD+QRAIN)·dm, so it includes rain water. It is not cloud water alone, and not the QCLOUD ≥ 1e-5 rule.
- Classes per cell × hour: both / CPU-only / GPU-only / neither. R = GPU-only / CPU-only, per case (summed over the three
  domains and 72 h) and pooled (sum over cases / sum over cases).
- Verdict rule: symmetric (pooled R in 0.8–1.25 and no case outside 0.5–2) → NOT_A_BUG; otherwise FINDING with its size.
- Diagnostics only (not in the verdict): an X-only cell-hour counts as a "hit" if the other side has an event at the same
  cell within ±1 h, at the same hour within ±1 cell (3×3), or within ±1 h and ±1 cell. Mean first-onset shift = mean over
  cells active on both sides of (first GPU event hour − first CPU event hour). The other side's value at X-only cells is
  binned as a fraction of the threshold.
- A case with no events on either side has R = 0/0. It is reported as undefined, not as outside the band.
- **Non-finite values (readback rule, E200):** `rain_rate_mm_h` must be all-NaN at τ0 (`np.isnan`, as in the cf54 readback;
  a finite value or ±Inf there is rejected) and finite at τ1–72. `LWP` must be
  finite everywhere. Any other non-finite value rejects the case pair (it raises; nothing is classified silently). Each
  extract's READBACK.json must have no failures, 0 foreign non-finite values and 73 frames. tau_h must be 0..72, lead_h =
  tau_h+6, and actual times and XLAT/XLONG must be equal on both sides. Built-in controls: one NaN, +Inf or −Inf at τ5 in
  either field, and a finite value, +Inf or −Inf in the rain rate at τ0, are all **REJECTED** (9/9); the healthy control is
  accepted [M]. (Fix after cr-w43-conv FIX 22:32Z: the first version tested only for finite values at τ0 and so accepted
  ±Inf there; the real inputs are all-NaN at τ0, so no count changed.) No real
  input was rejected. `rain_negative_flag` is 0 everywhere.

## (a) rain_rate ≥ 0.1 mm/h — FINDING by the frozen rule [M]

| case | both | CPU-only | GPU-only | GPU/CPU-only | X-only hit ±1 h (C/G) | ±1 cell (C/G) | ±1 h ±1 cell (C/G) | mean 1st-onset shift h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wn3_20260120 | 11288 | 4916 | 5108 | 1.039 | 1543/1592 | 1713/1671 | 2972/2943 | +1.51 |
| wn3_20260220 | 0 | 0 | 0 | undef (0/0) | 0/0 | 0/0 | 0/0 | – |
| wn3_20260227 | 17047 | 3435 | 3445 | 1.003 | 1429/1647 | 1757/1903 | 2517/2663 | −0.00 |
| wn3_20260502 | 160 | 13 | 9 | 0.692 | 9/5 | 13/7 | 13/7 | +0.15 |
| wn3_20260608 | 191 | 10 | 3 | **0.300** | 7/3 | 10/3 | 10/3 | +0.05 |
| wn3_20260614 | 629 | 726 | 724 | 0.997 | 95/99 | 98/135 | 214/229 | +0.05 |
| era5_20240218 | 5 | 88 | 62 | 0.705 | 14/15 | 1/0 | 26/20 | +0.18 |
| era5_20240302 | 5728 | 2540 | 2841 | 1.119 | 1103/1158 | 971/1015 | 1740/1938 | +0.34 |
| **pooled** | 35048 | 11728 | 12192 | **1.040** | 4200/4519 | 4563/4734 | 7492/7803 | +0.60 |
| twin 0227 (CPU vs CPU) | 17046 | 3436 | 4236 | 1.233 | 1594/2063 | 1698/2094 | 2631/3324 | +0.28 |

The pooled R (1.040) is inside 0.8–1.25. **The rule trips on one case: 0608, R = 0.300 (3 GPU-only vs 10 CPU-only cell-hours).**

Size of the finding:
- **0608:** 13 asymmetric cell-hours out of 204 event cell-hours. Rain in this case peaks at only 0.18 mm/h (d01) and
  0.22 mm/h (d03), so every event is close to the threshold.
  - 9 of the 10 CPU-only cell-hours have GPU 0.08–0.1 mm/h; the 10th has GPU < 0.01 mm/h.
  - All 3 GPU-only cell-hours have CPU 0.08–0.1 mm/h.
  - All 13 have an event on the other side within ±1 cell, and 10/13 within ±1 h.
  - Mean first-onset shift +0.05 h (40 cells, all within ±1 h). All 13 are in the 0.1–0.5 mm/h class.
- **Pooled excess:** GPU-only − CPU-only = **+464 cell-hours** (+4.0 % of CPU-only).
- **Pooled mean first-onset shift: +0.60 h** (GPU later; 16,053 cells). It is dominated by 0120 (+1.51 h, 5,835 cells).
  Other cases range from −0.00 to +0.34 h. The CPU twin of 0227 gives +0.28 h.
- **By rain-rate class (pooled, CPU-only / GPU-only):**

  | rate (mm/h) | CPU-only | GPU-only | both |
  |---|---:|---:|---:|
  | 0.1–0.5 | 9995 | 10583 | 19991 |
  | 0.5–1 | 1124 | 1007 | 5374 |
  | 1–2 | 450 | 476 | 4168 |
  | 2–5 | 130 | 114 | 3962 |
  | 5–10 | 29 | 12 | 1323 |
  | ≥ 10 | 0 | 0 | 230 |

  The "both" column is binned by the CPU rate. Only the 5–10 mm/h class leans toward CPU-only (29 vs 12, 41 cell-hours
  out of 23,920 asymmetric). Below 5 mm/h the classes are balanced within ±13 %.
- **Near-threshold share (pooled):** at CPU-only cells the GPU value is 0.08–0.1 mm/h in 1505 cases and below 0.01 mm/h in
  4819. At GPU-only cells the corresponding counts are 1534 and 4928. The two directions match.
- **0220:** no cell-hour reaches 0.1 mm/h on either side (max CPU 0.068 / GPU 0.068 mm/h in d03), so its R is undefined.

## (b) LWP ≥ 0.01 kg/m² (proxy) — NOT_A_BUG by the frozen rule [M]

| case | both | CPU-only | GPU-only | GPU/CPU-only | X-only hit ±1 h (C/G) | ±1 cell (C/G) | ±1 h ±1 cell (C/G) | mean 1st-onset shift h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wn3_20260120 | 48567 | 18212 | 17325 | 0.951 | 8056/7609 | 8238/7894 | 13635/12501 | +0.51 |
| wn3_20260220 | 85379 | 1762 | 2617 | 1.485 | 1358/2062 | 1232/2009 | 1698/2456 | +0.02 |
| wn3_20260227 | 101417 | 14283 | 11916 | 0.834 | 9460/7165 | 8547/7548 | 12834/10537 | −0.46 |
| wn3_20260502 | 4102 | 997 | 1306 | 1.310 | 665/671 | 643/844 | 915/1123 | +0.07 |
| wn3_20260608 | 67157 | 2402 | 3603 | 1.500 | 1624/2611 | 1787/2839 | 2206/3406 | −0.04 |
| wn3_20260614 | 146841 | 16357 | 14807 | 0.905 | 9110/8393 | 7867/7088 | 12903/11784 | +0.06 |
| era5_20240218 | 42456 | 11836 | 11312 | 0.956 | 6186/5977 | 5160/4938 | 9317/8705 | +0.41 |
| era5_20240302 | 46085 | 19935 | 20030 | 1.005 | 10874/10122 | 9239/8965 | 16142/15590 | +0.08 |
| **pooled** | 542004 | 85784 | 82916 | **0.967** | 47333/44610 | 42713/42125 | 69650/66102 | +0.06 |
| twin 0227 (CPU vs CPU) | 103687 | 12013 | 11169 | 0.930 | 7237/6959 | 7955/7548 | 10679/10142 | −0.36 |

Pooled R is 0.967 and all cases lie within 0.834–1.500, so the result is symmetric. Disclosure: the rule is per case,
summed over domains. Some single domains with small counts lie outside 0.5–2: 0220 d01 12.25 (98 vs 8), 0614 d01 3.32
(93 vs 28), 0608 d02 3.26 (388 vs 119), 0220 d02 2.62 (257 vs 98). Per-domain values for every case are in the JSON.

## Twin noise reference
- **0227: available [M].** The ALISIOS-matching CPU twin (`<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu`: same
  executable and inputs, 4×1 instead of 2×2 decomposition, END_PROOF PASS, 73 frames per domain) was extracted with the
  teacher's own extractor at the version that produced the reference extracts. That is `analysis/teacher_extract.py@cf54a11a42`,
  whose SHA 6d44e799 equals `producer_code_sha256` in the manifests; HEAD ab805062 differs only in readback reporting.
  - Readback: 0 foreign non-finite values, 73 frames.
  - Static fields (XLAT/XLONG/HGT/LANDMASK) and τ0 LWP/RAINNC/T2 are bitwise equal to the reference extract on all three domains.
  - Twin vs reference R: **rain 1.233** (domains d01 1.037 / d02 1.599 / d03 1.199), **LWP 0.930** (d01 1.389 / d02 0.889 /
    d03 0.931). GPU vs reference for 0227: rain 1.003, LWP 0.834. One twin realization only; no noise interval is defined,
    so these ratios are a reference, not a band.
- **ERA5 0302: UNAVAILABLE.** w43-teacher's cleanup at 2026-10-21T21:18Z deleted the 2×2 CPU twin wrfouts (MAILBOX). Only
  per-lead aggregate difference statistics remain (`w43-teacher/score/C_era5_20240302/x`), and they contain no cell-level
  fields. No twin is invented.
- Twin extract and log: `<USER_HOME>/wrf_gpu2_lanes/w43-conv/contingency/twin0227_cpu/` (kept for the critic).

## Interpretation [I] (not part of the verdict)
By the frozen rule, rain is a FINDING. It rests on 13 cell-hours in one near-dry case (0608). In that case every event lies
within 0.12 mm/h of the threshold, and every asymmetric cell-hour has an event on the other side within one cell. A CPU-vs-CPU
decomposition change on 0227 already moves R by 23 %. I do not conclude a bug from this, and I propose no threshold change.
The direction-consistent signals worth noting are small:
- 5–10 mm/h: CPU-only 29 vs GPU-only 12;
- 0120 first-onset shift +1.5 h.

## Limits
- LWP proxy as disclosed above.
- d01/d02 are teacher crops, not whole domains.
- The cell-hours are strongly correlated, so there is no significance test, as frozen.
- The extracts are the teacher's inputs and were not re-derived from GPU wrfouts. The GPU wrfouts are deleted; the
  teacher's readback seals were checked.
- The ±1 h / ±1 cell hit counts are diagnostics only.
