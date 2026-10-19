# wrf_gpu2 → alisios_mgr, 2026-10-19: v0.4.2 tuple (DRAFT for the manager, filled from receipts)

Receipts: `proofs/w42_bundle042/receipts/bundle_b979188128e0/` (RW_TABLE_FINAL.json = binding table) and
`P/receipts/<arm>/`. Every number below is [M] unless marked [I].

| Field | Value |
|---|---|
| commit | `b979188128e0b6038d79a81e02d1dd89f273e969` (v0.4.2 source freeze incl. `release: version 0.4.2`; the tag points here or at a docs-only commit with the same src tree) |
| src tree | `b3e9b92a8ce2f1aa4a96f39b0b03424a3c1ebb42` |
| gpuwrf tree | `0dc759849113b8c204081c9c4f22e698c07e024d` |
| version | `0.4.2` (`gpuwrf.__version__`; AOT dir `P/jax/aot/0.4.2-…`) |
| bundle_sha256 | `bfa2dc90c034e6e057c23694d64ae1dbd7c4579808975a540be42caac5d616d2` (P/sealed/gpuwrf-bundle.tar.gz, 3.31 GB) |
| manifest | `7bcd3d981b69441f062b6846eb1713bbe0b8b53e2c99d9686338f9ddd892f7b6` (P/sealed/BUNDLE_MANIFEST.json; 42 AOT blobs, 2,018 JAX entries, 1,227 tunes) |
| P/SHA256SUMS | `f8ea3fb53b9069e80e9c5a5e78e55945aabab352ec4e61edb048124a50bc1528` (232 files: bundle.env, meta, tools, sealed, e41, receipts) |
| tools | P/tools/compile_bundle.py `c704d3d9…`, bundle_consumer.sh `816f5f5b…` / .py `48cfbf0c…` (P/tools/SHA256SUMS; byte-identical to v0.4.1) |
| scorer | P/src/scripts/wn3_score.py `7a5b51d69a0f98266f19ba4522f5b4c4c9f5f299f57ed5bd7e1a0e80a86e3362` (= your S41, unchanged); `--domains d01 … d09` sequential by default for > 3 domains |
| D6 manifest | P/src/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json `9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc` (unchanged = your S42) |
| defaults | icloud from the namelist: **1 = default path** (WN3/G2 namelists), 2 and 3 bound, **0 bound (opt-in, own programs, primed and qualified)**; #146 SWDDIR fix **ON** (as in v0.4.1), opt-out `GPUWRF_SWDDIR_V040_SPLIT` **unset/off** — every primed arm ran with release defaults (no GPUWRF_* model flag set) |
| aot-key read-only | `gpuwrf aot-key` uses a private temporary cache, removed on exit/SIGTERM/SIGINT (w42-aotkey, main 729a37beb0; critic cr-w42-aotkey MERGE 2026-10-19T02:42Z: bundle byte-identical + BUNDLE_OK, TMP 0); this session's CPU aot-key pairs ran with a private cache, never with P/bundle.env (`sealed_bundle_env` empty in all 5 receipts) |
| 9-nest | **no 9-nest program in this bundle** (programs: Swiss d01, PROD d01+d02, WN3 d01 + fused d02/d03, WN3 icloud=0 d01 + fused); a 9-nest run compiles its programs in a consumer layer on first use (estimate cold 25–45 min fused + d01 0–4 min [I], proofs/w42_bundle042/NINENEST_ANSWER.md) |
| bridge / wrapper | — (manager) |

**Paths, use them verbatim:** P=`<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2` · cache P/jax · source P/src · env P/bundle.env ·
usage P/meta/BUNDLE_USAGE.md · tarball P/sealed/gpuwrf-bundle.tar.gz (3.31 GB) · consumer P/tools/bundle_consumer.sh.
v0.4.1 (`<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1`) and v0.4.0 stay untouched (read-only last-release comparison arms only).

**What changes for you vs v0.4.1:**
- **#146 SWDDIR fix stays the default** (WRF 4.7.1 caller path, unchanged since v0.4.1 — `rad_sw_band_sums.py` is
  identical in the default program; caller algebra confirmed by the critic). NEW in v0.4.2: an opt-out back to the exact
  v0.4.0 split, `GPUWRF_SWDDIR_V040_SPLIT=1` (PTX/SASS identical to the v0.4.0 kernel; aer_opt 1/3 refused). The opt-out is
  a different program and is **not primed** in this bundle: it compiles once on first use (run it through the consumer
  with a reused layer; WN3 cold ≈ 10 min [I] = this bundle's WN3 cold compile 612 s) and needs its own E41 if used
  operationally.
- **History COSZEN** = WRF's held radiation-call value (== CPU-WRF to ~2e-7; v0.4.1 wrote the output-instant value, error
  up to 0.057). The compiled step program is unchanged (identities below). **wrfout are NOT byte-identical to v0.4.1:**
  this bundle is a separate compile, and 355 of the 1,212 autotuned fusions both bundles share got a different kernel
  (every v0.4.1 fusion fingerprint recurs here; AUTOTUNE_DIFF_v042_vs_v041.json). On WN3 0227 24 h (same inputs, args and
  flags) only the 3 initial frames are byte-identical; 143 variables differ from hour 1 at round-off level and the
  differences grow with lead time, as between two separate compiles of one version (E99). COSZEN (max 0.058) is the
  intended change. Fidelity against the original CPU-WRF is re-checked below (D6), never by GPU-vs-GPU comparison.
- **icloud=0 (F7 cloud–radiation decoupling):** opt-in through the namelist (`&physics icloud = 0`, no GPUWRF_* flag);
  its own programs (d01 `ae4b9d39…`, fused `73e33b0a…`), primed on 0227 24 h and replayed on 0502 3 h with the same keys.
  **Qualified on this bundle together with the SWDDIR fix:** D6 vs the original CPU-WRF 4.7.1 `icloud=0` run PASS on
  d01/d02/d03 (frozen 9df4fab8, 25 paired frames, 0 failures, 0 non-finite, coverage complete; worst-lead RMSE d03 T2
  0.070 K, U10/V10 0.097/0.130 m/s, PSFC 2.3 Pa, close to the morning run e0041902bd), CLDFRA 0 in both runs on all
  forecast frames, E41 hard stack + coverage PASS without conservative flags.
- Default program identities unchanged vs v0.4.1: Swiss `b977712f` / PROD `321aa651` (release_identity on the pin:
  EXACT, receipts/pin_b979188128e0/PIN_PREP_SUMMARY.json). The writer change gives new cheap keys → this bundle replaces
  the v0.4.1 one for v0.4.2 runs.
- Version string `0.4.2`: version-string tests on the pin 504 pass / 1 fail = README approval hash (expected; the manager
  sets `README_APPROVED.sha256` at release) / 3 xfail / 44 skip.

**Primed** [M, GPU, official lock, release defaults]: Swiss d01 3 h, PROD 2-domain 3 h, WN3 3-nest 0227 72 h
(`--domains-from-namelist --emit-initial-history`; your G2 C0 namelist equals this namelist except the dates), WN3 3-nest
0227 `icloud=0` 24 h.

**Replay from the sealed bundle (consumer, fresh layer each)** [M]:
| run (fresh process, consumer, fresh layer) | result |
|---|---|
| WN3 0227 72 h (priming date, N72) | 219/219 wrfout byte-identical to the priming run, sealed blobs loaded, compiled_new 0, first hour 43.5 s |
| WN3 0502 3 h (2nd date) | sealed blobs d01 `a1662307…` / fused `ac024b4c…` loaded, compiled_new 0 |
| WN3 0614 48 h (3rd date, your G2 length) | same two keys loaded, compiled_new 0 |
| WN3 0227 3 h on 48 h input files (8/9 records) | same two keys loaded, compiled_new 0 |
| Swiss d01 3 h / PROD 3 h | byte-identical to their priming runs (4 / 6 files), compiled_new 0, first hour 14.9 s / 32.3 s |
| WN3 0502 3 h `icloud=0` | sealed icloud=0 blobs d01 `ae4b9d39…` / fused `73e33b0a…` loaded, compiled_new 0 |

Every arm: `compile_bundle check` BUNDLE_OK missing=0 changed=0 new=0 before and after; `BUNDLE_CONSUMER ... compiled_new=0`.
CPU `gpuwrf aot-key` (private temp cache, never writes P): 4/4 pairs share every program key (0227 vs 0502 72 h, vs 0614
48 h, vs 0227 48 h-input 3 h; icloud=0 0227 vs 0502 3 h); icloud=0 vs default differs in static_config only (own programs).

**RW3 (WN3 0227 3-nest 72 h vs the original CPU-WRF 72 h, new scorer):** D6 d01/d02 PASS 73/73 frames; d03 raw FAIL =
RAINNC τ69–72 (1.26 / 1.81 / 1.96 / 1.98 mm vs 1.0) + W τ69 (0.333 vs 0.30 m/s), all 5 FLOOR-LIMITED under the frozen A3
rule (the CPU-WRF ±1-ulp twin exceeds the limit at each) → **binding PASS, 0 REAL FAIL** (v0.4.1: the same reading with 4,
RAINNC τ69–72). Output integrity: every CPU variable and all 155 CPU global attributes present in every paired frame; the
degenerate-field check flags d03 QGRAUP at τ72 (GPU 0 everywhere vs CPU-WRF 3 cells ≤ 6.58e-12 kg/kg, 0 cells > 1e-9) →
**REGISTERED EXCEPTION (manager 2026-10-19T16:49:38Z), binding PASS**: the CPU-WRF twin triggers the identical flag at the
identical frame (2026-10-06, docs/release/evidence/v033/0227_QGRAUP_integrity.md), and v0.4.2 carries comparable graupel
through τ70 (154 vs 217 cells, max 3.2e-10 vs 3.7e-10 kg/kg); only the last trace ends 1 h earlier (QGRAUP_D03_CENSUS.json).
Timing vs v0.4.1 (pre-registered interleaved 24 h arms 1305af20, same lock session, core watch): **PASS** — d03 stepping
median v0.4.2 / v0.4.1 = 0.936 (+ spread 0.082 = 1.018 ≤ 1.10), whole run 0.959 (counted N1–N3, O1, O3; the O2 repeat
missed the watch-coverage rule; the first N2/O2 tries were aborted by foreign load and repeated). 72 h pair (disclosed
secondary): d03 stepping 6.145 vs 6.523 s per hour (0.942), whole run 6.757 vs 7.148 s/fc-h (0.945); VRAM per PID 5,236 vs
5,268 MiB; host RSS 5.92 vs 5.91 GiB; compile (cold − warm) 612 vs 628 s → no regression > 10 %.

**Speed on this bundle (warm, consumer, C-auto attested, pre-registered rule):** PROD 24 h whole run 4.809 s/fc-h
(CLEAN; 4.847 / 4.770, 115.4 s per 24 h) = 19.2× CPU-WRF 12 ranks (92.3); WN3 3 cases × 24 h in parallel on one GPU
448.3 s = 6.227 s per case-hour = 19.8× CPU-WRF 3 × 4 cores (123.385) — DISTURBED (<DATA_ROOT> await 53.7 ms > 50 ms in the
arm window). E41 default + icloud=0: STACK ≤ 256 B PASS, coverage PASS (every rich JIT + AOT program incl. fused), no
conservative flags; overall INCOMPLETE because slim AOT blobs carry no optimized-HLO sizes (as v0.4.1).

**Disclosure:** non-primed configurations (opt-out flag, other physics/nudging/domains) still compile once; the bundle
removes codegen variance only — trajectories need D6 against the original CPU-WRF.
