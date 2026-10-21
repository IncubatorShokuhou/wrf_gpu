# v0.4.3 release plan (lane w43-release, 2026-10-21)

Owner goal: v0.4.3 = v0.4.2 numerics + ALISIOS-reported fixes/qualification, validated and released as fast as possible.
ALISIOS acceptance (cross `TO_WRF_GPU2_ACCEPTANCE_V043_20261021.md`) item 1: fixed identities as SHA, no silent engine switch.
Manager 2026-10-21: ALISIOS accepts "v0.4.2 numerics = v0.4.3" ONLY if the proof binds the SHAs of the COMPILED numerics
(bundle artefacts), not just the source tree, and shows that the only diff (host_suspend in receipts) has no numeric effect.

## 0. Candidate

| item | value |
|---|---|
| v0.4.2 release | tag v0.4.2 -> `5c34b4189d`, src tree `b3e9b92a8c`, bundle `<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2` (pin `b979188128`, bundle_sha256 `bfa2dc90…`) |
| v0.4.3 pin | lane/w43-release `da7a8acbf5` = main `218e27ea28` + version bump; src tree `6f98d8d855` |
| src diff v0.4.2 -> pin | `src/gpuwrf/cli.py` (+15/-1: `SuspendWatch` created in `_cmd_run`, `payload["host_suspend"]`, failure note, stderr warning), NEW `src/gpuwrf/runtime/host_suspend.py` (51 lines, reads `CLOCK_BOOTTIME - CLOCK_MONOTONIC`), `src/gpuwrf/__init__.py` (`__version__` 0.4.2 -> 0.4.3). Nothing else under `src/`. |
| non-src diff | kernel/.agent files, proofs/v043 analysis (merge convention), pyproject + docsite version |

If another traced-code merge lands before the pin is final, this plan is re-run on the new pin (identity section C0 decides
whether the carry-forward still holds; any traced change => sections C2/C3 are expected to FAIL and the full RW3 D6 runs).

## 1. Compiled-numerics identity proof v0.4.2 bundle -> v0.4.3 bundle (pre-registered, before any v0.4.3 GPU number)

Why it is not automatic: (a) the AOT cheap key contains `gpuwrf_version` (aot_cheap_key.py `fn_identity`, :802) and the
AOT directory is version-keyed (`aot/0.4.x-jax0.10.0-jaxlib0.10.0-cuda_sm120`), so v0.4.3 keys/paths differ by
construction; (b) Pallas/Triton custom calls embed the absolute source path in the lowered HLO (E58/E109) and the bundle
source lives at `P/src` (different path per version), so lowered-HLO SHAs (`AotMeta.hlo_sha256`) differ; (c) cold compiles
are not run-to-run deterministic (E99: the v0.4.2 compile chose a different kernel than v0.4.1 in 355 of 1,212 shared
autotuned fusions) — a plain cold compile of identical source would give round-off-different outputs.

Mechanism that makes identity possible: the v0.4.3 bundle is compiled with the v0.4.2 bundle's per-fusion autotune results
copied in BEFORE priming (`P42/jax/xla_gpu_per_fusion_autotune_cache_dir`, 1,227 files -> `P43/jax/…`; JAX forces that
dir under the JAX cache, E35). Autotuning only selects among correct kernels (numerically inert by design,
runtime/xla_autotune.py); seeding makes v0.4.3 pick exactly v0.4.2's kernels. Precedent: w41-bdyrec seeded from the v0.4.0
bundle (Swiss frames byte-equal to the release bundle). Pre-measured on the v0.4.2 bundle [M, CPU,
`native/v042_native_ids.json`]: the bundle path occurs 0 times in Riegeli records 0–15 (record 2 = the native cubin) and 0
times in the optimized HLO of every program; only 2 occurrences per blob elsewhere (compile options) -> exact record-2 SHA
equality across bundles is achievable.

Checks (each per program; programs = AOT `d01`, `d02`, `fused_d02` (fused d02+d03), `init_solve_rrtmg_lw_column`,
`init_solve_rrtmg_sw_column` in both bundles, plus the rich JIT entries of the JAX cache):

| id | check | tool | PASS rule |
|---|---|---|---|
| C0a | traced program identity on CPU | `proofs/s1_port/release_identity.py` on the pin | Swiss `b977712fe325…` / PROD `321aa651067f…` EXACT — **[M] PASS 06:2xZ** (`receipts/pin_da7a8acbf5/release_identity_{swiss,prod}.json`) |
| C0b | cheap-key COMPONENTS v0.4.2 src vs v0.4.3 pin (`gpuwrf aot-key --json`, CPU, same case dirs: WN3 0227 72 h vs 0502; 0227 icloud=0 vs 0502 icloud=0) | `tools/aotkey_pair.sh` + `aotkey_compare.py` | every component equal except the ones that carry `gpuwrf_version`; each differing component explained field by field |
| C0b' | cheap-key source closure + fn_identity counterfactual | `tools/closure_diff.py` | **[M] PASS 06:2xZ** (`receipts/aotkey/CLOSURE_DIFF.json`): trace closure 519 files in both trees, ONLY `__init__.py` differs (the version line); `cli.py` / `runtime/host_suspend.py` NOT in the closure; v0.4.3 `fn_identity` with `__version__` reverted to 0.4.2 == v0.4.2 `fn_identity` (81a67c2c…) |
| C0c | host_suspend has no path into traced code | source read + import graph | `host_suspend` imported only inside `cli._cmd_run`; not in the cheap-key source closure; only writes `payload["host_suspend"]` + stderr |
| C1 | optimized HLO (decoded from each AOT blob) | `tools/native_ids.py` | SHA equal per program (path-normalized SHA reported too; raw expected equal since 0 path hits) |
| C2 | native executable: Riegeli record 2 (the cubin) SHA and every other record SHA | `tools/native_ids.py` | record-2 SHA equal per program; all records equal except those carrying the bundle path (listed; expected none) |
| C2b | no new autotune entries | file count of `P43/jax/xla_gpu_per_fusion_autotune_cache_dir` after priming | == 1,227 (every autotuned fusion was a v0.4.2 hit) — disclosed either way |
| C2c | program set | AOT dir listing + `last_cheap_key` | same program families and alias counts as v0.4.2 (d01 4, d02 1, fused_d02 2, init LW 7, SW 7) |
| C3 | behaviour: wrfout whole-file SHA of every v0.4.3 arm vs the v0.4.2 bundle receipt of the same arm (`P42/receipts/<arm>/wrfout_sha.txt`; the writer embeds no version/path, wrfout_writer.py:2596/2618) | `tools/bytes_vs_v042.py` | A1 Swiss 3 h, A2 PROD 3 h, A3 WN3 0227 72 h (219 files), A4 0227 icloud=0 24 h, B2–B8 replays: all files byte-identical |
| C4 | RW1 within v0.4.3 | warm replay of A3 (N72) | 0 new compiles, 219 files == A3 |

Decision rule (pre-registered):
- C0–C3 all PASS => "v0.4.3 compiled numerics == v0.4.2" is PROVEN at the artefact level; the v0.4.2 numerical validation
  carries forward unchanged (RW3 D6 72 h + QGRAUP t72 registered exception, icloud=0 D6 3/3, E41, teacher/ALISIOS tables
  computed on the v0.4.2 bundle) — per AGENTS "Validation reuse" (no repeat for proven byte-identical changes). RW3 then
  only adds timing/VRAM/RSS vs v0.4.2.
- C2 differs but C0 PASS => NOT identical: v0.4.3 is "source-identical, separately compiled" (round-off-different). Then
  the full RW3 (D6 72 h vs original CPU-WRF + integrity) runs on v0.4.3 and the ALISIOS qualification must name the
  v0.4.3 bundle SHA; the carry-forward claim is NOT made. Reported as such, no reinterpretation.
- C3 differs while C2 is equal => STOP and diagnose (would mean host-side output difference).

## 2. GPU session (ONE lock session; manager slot; template = w42-bundle042 tools, frozen copy)

Order (expected minutes from the v0.4.2 actuals): SEED autotune (CPU, seconds) -> PRIME A1 Swiss 3 h (5.5) -> [early CPU
C2 check on d01 Swiss, logged, non-blocking] -> A2 PROD 3 h (9.5) -> A4 0227 icloud=0 24 h (~8) -> A3 0227 72 h cold (19)
-> SEAL + check BUNDLE_OK (1) -> REPLAY via consumer: B2 0502 3 h, B4 0227 48 h-input 3 h, B5 Swiss 3 h, B6 PROD 3 h,
B8 0502 icloud=0 3 h, B3 0614 48 h (~12) -> RW3T: N72 (v0.4.3 72 h warm = RW1) + interleaved N1 O1 N2 O2 N3 O3 24 h
(N = v0.4.3, O = v0.4.2 bundle via its consumer) + O72 (~45). Total ≈ 100–110 GPU-min. Core watch / DISTURBED rules as v0.4.2.
CPU follow-up: E41 (`scripts/wave_cubin_gate.py`, every rich JIT + AOT program incl. fused) BEFORE any cleanup, C1/C2
native_ids on P43 vs P42, C3 bytes, aot-key RW4 pairs on the pin, RW3T verdict, RW table.

Cases: own copies `<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases` — 0227/0502/0614/0227_48h byte-identical to the v0.4.2
case list (`cases_sha256.txt`, 32/32 equal), 0227 icloud=0 namelist reconstructed byte-exact (sha `2c15e10b…`, = w42-icloud0
spec), 0502 icloud=0 = one-line edit. Sources: w43-teacher / w41-bdyrec staged copies (0 new ALISIOS reads).

## 3. Release documents, final.sh

Release notes v0.4.3 + support-matrix delta + KNOWN_ISSUES (#148 INCONCLUSIVE, #149, coastal wind NOT_A_BUG/A NOT_ENGINE,
0531 truncated = not a 72 h PASS, host suspend receipts), README only if needed (lint + manager approval sha).
`scripts/release/final.sh --version 0.4.3 --label final …` dry path + wheel smoke prepared; manager tags/pushes.

## 4. Status

| step | status |
|---|---|
| inventory + version bump | DONE (da7a8acbf5) |
| C0a identities | PASS [M] |
| C0b' closure + fn_identity counterfactual | PASS [M] |
| C0b aot-key components per program | PASS [M] (receipts/aotkey2/COMPONENTS.json: 16/16 records differ only in fn_identity, source_fingerprint and their digest program_key) |
| C0b'' pre-bump main 218e27ea28 (v0.4.2 src + host_suspend) | PASS [M]: cheap keys + all components IDENTICAL to v0.4.2 for 16/16 records -> host_suspend has no compile-key effect |
| cases staged | DONE |
| GPU tools fork + dry test | DONE (lifecycle, rc4 early / rc6 late refusal, TERM teardown 0 scopes) |
| GPU session 13:05–14:57Z real | DONE rc 0, 112 GPU-min, 0 hibernation lines; pin `888804310bb6` (src tree `6f98d8d8558a` = `da7a8acbf5`) |
| seal | bundle_sha256 `d2f4f197b8affdecc0c297faae3593eb9d4df658cc3a58fc24f6523518aeee70`, gpuwrf tree `3e3642f5a261`; BUNDLE_OK after seal, before/after all 6 replays, after E41, final |
| C1/C2 native identity | **PASS [M]** all 21 programs (d01 4/4, d02 1/1, fused_d02 2/2, init LW 7/7, SW 7/7): optimized-HLO SHA, record-2 cubin SHA and ELFs equal to v0.4.2 bfa2dc90 (`receipts/followup/NATIVE_IDENTITY.json`, both bundles scanned with the same frozen tool) |
| C2b autotune | **PASS [M]** 1,227 == v0.4.2 seed, 0 new/changed/missing |
| C3 bytes | **PASS [M]** 10/10 arms, 497/497 wrfout byte-identical to the v0.4.2 bundle receipts (A3 72 h 219, A4 icloud=0 24 h 75, B3 0614 48 h 147, …) |
| C4 RW1 | **PASS [M]** N72 (warm 72 h) 219/219 == A3, compiled_new 0 |
| O72 control (v0.4.2 replay vs v0.4.2 A3) | **PASS [M]** 219/219 (top-up 17:01–17:10Z) -> `receipts/followup_id2/IDENTITY_VERDICT.json` = **IDENTICAL** (C0a, C0b, C1/C2, C2b, C3/C4 all PASS): v0.4.2 numerical validation carries forward |
| E41 | INCOMPLETE = identical to v0.4.2's own gate (compact AOT HLO kInput UNKNOWN, E201); STACK PASS max 32 B; same 7 ELFs |
| RW2/RW4 | PASS (6 replays, 0 compiles, BUNDLE_OK; CPU aot-key pairs SHARED) |
| RW3 fidelity | carried by C3 (A3 bytes == v0.4.2 A3: D6 binding PASS + QGRAUP t72 registered exception) |
| RW3 speed | Frozen RW3T verdict **INCONCLUSIVE** (top-up 17:00–17:36Z same session; counted N2 N3 O3 only, others failed watch_cover = sampler slip; counted N/O 0.956) — disclosed as is. **Manager ruling 2026-10-21 ~17:50Z: NO REGRESSION** (rule announced 16:47Z: all 6 same-session 24 h arms within 10 %, N/O 0.958, prereg formula 0.958 + 0.120 = 1.078 <= 1.10; compiled numerics IDENTICAL), not a STOP. Session-1 cross-session flag (N72 8.67 vs 6.145 s/h) = host load (O72 6.03 s/h in the top-up). VRAM 5268 vs 5236 MiB, RSS 6.20 vs 5.92 GiB |
| disk (manager DISK URGENT) | every finished arm's wrfouts deleted after receipts (`receipts/arm_receipts/`, incl. top-up); lane keeps N72 21 GB (delete after critic) |
| bundle receipts | P/receipts 164 files + P/meta + P/e41/icloud0; P/SHA256SUMS 233 lines sha256 `37941c9a61ac…` (v0.4.2 scope); BUNDLE_OK + sha256sum -c OK (`receipts/BUNDLE_SHA256SUMS_v043.txt`) |
| RC tests | RC65 (proofs/v043/w43-reds/FINAL_RC_FILES.txt) 1990 P / 15 S / 0 F measured by w43-reds on 49746fcc33; src/tests/scripts/data/pyproject trees of the content commit are byte-identical to it -> transfers (validation reuse); the 2 reds-fixed files re-run live on the content commit (READY line) |

## 5. Critic packet (READY-FOR-CRITIC; critic cr-w43-teacher)

Gate = this plan: section 1 decision rule (pre-registered) + real-world gate RW1–RW4 + release docs + final.sh.

| item | where |
|---|---|
| identity verdict IDENTICAL | `receipts/followup_id2/IDENTITY_VERDICT.json` (+ `BYTES_VS_V042.json`, `VIEW.txt`); native `receipts/followup/NATIVE_IDENTITY.json`, `native_ids_v04{2,3}.json`; autotune `AUTOTUNE_SEED_CHECK.json`; C0 `receipts/pin_da7a8acbf5`, `receipts/aotkey*/`, `receipts/aotkey/CLOSURE_DIFF.json` |
| RW1–RW4 | `receipts/followup/RW_VERDICT.json`, `RW3T_*.json` (session 1), `receipts/followup_topup/RW3T_*.json` (top-up), arm receipts `receipts/arm_receipts/{gpu,rw3t,topup}/*/WRFOUT_FILES.tsv` |
| E41 | `receipts/followup/e41/wave_gate.json`, `e41_icloud0/`, compared with the v0.4.2 bundle's own gate |
| bundle | `<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3` (bundle_sha256 `d2f4f197…`, P/SHA256SUMS `37941c9a…`), `BUNDLE_USAGE.md` |
| tools | frozen GPU tools `$L/tools_b48c141de9` (record), follow-up fix copy `tools_b48c141de9_fu2` (only P43 line), top-up `tools_topup` (topup_session b3973148, waiter 2684efcc), disk relief v1/v2, assembler; copies in `tools/` |
| docs | `release_notes/RELEASE_NOTES_v0.4.3.md`, `docs/release/SUPPORT_MATRIX_v0.4.3.md`, `docs/release/USER_GUIDE_v0.4.3.md`, `KNOWN_ISSUES.md` (v0.4.3 sections), `CHANGELOG.md`, `README.md` (lint PASS, SHA for manager approval), docsite (synced + built) |
| consistency | release-notes status table == qualification note v1 §4 (cross FROM_WRF_GPU2_V043_QUALIFICATION_20261021T185448Z.md); teacher table cited from `proofs/v043/w43-teacher/TEACHER_QUAL.md` |
| disclosed incidents | E163 aotkey re-run (receipts SUPERSEDED_E163), followup P43 bug (fu2 copy), O1 rc 124 at the 330 s budget (session 1), watch_cover exclusions (top-up), dry-only final.sh commit object 6018b6214d (never on a branch) |
