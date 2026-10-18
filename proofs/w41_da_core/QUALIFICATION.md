# w41-da-core qualification packet (A1 part 1: product LETKF + `gpuwrf da`, d01, opt-in)

Candidate: `lane/w41-da-core` HEAD named in the READY-FOR-CRITIC line; its code (src/tests) is identical to
snapshot **0eeb8fff1d** (critic FIX delta, next section). Original packet below: snapshot **987fbe8628** (= b01a98b1ef + main 72e8a3abda merged), then main 1b98beb9d3 merged as 03d1e8d4db (re-checks below); later commits touch only proofs/notes. Merges, not
rebases, because w41-da-ens/-cycle carry these commits. Base import: lane/a1-da 1131707318 (verbatim, 53a955450b).
All evidence is CPU (`JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, lane cores, nice 19, host load 44-80). No GPU used.

## Delta for critic FIX cr-w41-da-core 15:35Z (4 gates) — candidate 0eeb8fff1d

Code: c116e4d702 (items 1-4) + 61ab0781e7 / 575015c2ad (follow-ups found by the first delta run) + main 67ca9d2e1e
merged as **0eeb8fff1d**. All evidence below on the immutable snapshot of 0eeb8fff1d (clean, one 4G RAM slot,
lane cores, nice 19): `<USER_HOME>/wrf_gpu2_lanes/w41-da-core/delta_0eeb8fff1d/` (status.txt, suite/light junit +
logs, MUTANTS.json, e2e.log).

| Item | Fix | Regressions (tests) | Mutants (genuine assertion kills) |
| --- | --- | --- | --- |
| 1 template/valid-day coverage | `AlisiosPolicy.refine`: every ALISIOS file gets its COMPLETE valid-day coverage from its run's `namelist.input`, read in a guarded metadata session under the issue day (facade or fallback): `wrfinput` = its domain's start day (issue/valid rollover), `wrfout` = first frame .. `frames_per_outfile` history intervals (WRF Registry defaults 10 / unset interval -> run end), other case files = whole run; every coverage day + issue day is guard-scanned BEFORE any open/staging; an archive frame touching a non-ALLOW day is dropped, any other input refuses the plan; unknown coverage (no namelist, no start, no run length, frame outside the run) refuses. Both backends consume the refined plan (facade targets get the complete days) | `test_governance.py` 13 (rollover admitted under start day; DENY start day refused with 0 opens and no admit of that day; multi-frame coverage; later-frame DENY drops archive / refuses control; WRF defaults incl. unset frames_per_outfile; outside window; missing namelist; DENY issue-day run dropped before its metadata is requested), `test_alisios_facade.py` 9 (facade rollover template; parity; DENY before open in both backends) | M25 template under issue day, M26 file-name day only, M27 only issue days scanned, M28 no-namelist fallback to issue day, M29 frames default 1, M13 (now: DENY issue-day run must be dropped, not fail the plan) |
| 2 boundary width | `resolve_spec_bdy_width(geo, explicit)`: the case namelist value binds the analysis taper, the mean writer (new: refuses a mean that changes a zone cell) and the member writer (`write_member_cycle` resolves from the case geometry; the API passes the same width); explicit contradiction refused; no namelist and no explicit value refused in the API; namelist without the key = WRF default 5. Receipt/manifest record `{value, source, namelist}` (namelist relabelled to the original under the facade) | `test_letkf.py` (width 9 binds the taper, width-5 control shows the check is live, contradiction), `test_api_cli.py` (Swiss case with width 9 end to end: analysis + mean zone untouched at 9 rows, member writer receives 9; member writer refuses a row-6 change at case width 9 and a contradicting explicit 5; mean-writer zone refusal), unset-key unit test | M30 analysis ignores case width, M31 contradiction accepted, M32 member writer default 5, M33 mean zone check off, M41 unset key = unknown |
| 3 observation roles + cloud | API: `use = kinds & ~role_withheld`; role-withheld rows of an assimilated kind become verification rows; `analyse` additionally forces `role == 'withhold'` and cloud rows withheld (counters `qc.role_withheld`, `qc.cloud_not_assimilated`); config refuses `assimilate: ['cloud']` before reads, `withhold: ['cloud']` allowed (diagnostics kept) | `test_letkf.py` (mixed-role same-kind batch: analysis bitwise == the batch without the withheld row; cloud row: 0 assimilated, background bitwise), `test_api_cli.py` (one surface row marked withhold: calibration_obs = n-1, role_withheld 1, withheld_ok >= 1; cloud config refusal/withhold) | M34 API ignores role, M35 role rows lost from verification, M36 analysis ignores role, M37 analysis assimilates cloud, M38 config accepts cloud |
| 4 cap/to_cloud | REFUSED in this build (`moisture.supersaturation` keep only; config + `LetkfParams` refuse cap/to_cloud with the reason); supersaturation is a counted diagnostic; `rebalance_increment` refuses a changed hydrometeor (the fixed-QV rebalance carries only dQV). Pressure, boundary and water gates unchanged | `test_moisture.py` (keep-only, supersaturated analysis counted and unchanged, hydrometeor refusal), `tests/a1_da/test_analysis.py` (cap/to_cloud refused) | M39 cap/to_cloud accepted, M40 hydrometeor change accepted |

Results [M CPU, snapshot 0eeb8fff1d]:
- `tests/da` + `tests/a1_da` + `tests/test_docsite_links.py`: **335 passed, 12 skipped** (93 s; the 12 skips are the
  same absent a1-da real-data fixtures as before).
- Mutants: **41/41 KILLED, CONTROL_PASS** (M01-M41, sequential, 6 min).
- Light S1 consumers (CLI, catalog, namelist binding/check, NetCDF lock, contracts, mp fail-closed): **226/226**.
  The heavy S1 set is carried forward: the merged main code is other lanes' (critic-reviewed) and this delta
  touches only `gpuwrf/assimilation/**`.
- Real-data e2e (dev case 0227, facade + real guard): rc 0/0, admissible, 12 members, 108/108 sonde obs; admitted
  set, staged files (15), issue days (20260218..20260227, 5), valid days, spy counters and stage removal IDENTICAL to
  the 36e0 run; analysis wrfinput **333/333 variables bitwise** == 36e0 (the file SHA differs only in provenance:
  original ALISIOS paths instead of stage paths, `inputs_staged`, time); O-B/O-A identical (T 1.903 -> 1.239 K).
  The real run is itself a rollover: wg_20260227_18z_a1 starts 2026-02-28 00Z, so its template is now admitted under
  20260228 (already in the set via the control frame, hence the unchanged admitted day set). New receipt counters:
  role_withheld 0, cloud_not_assimilated 0, supersaturation (keep) 74,001 member cells above liquid saturation,
  max RH 1.58, counted and not changed (not separated from background supersaturation). Analyse 17.5 s, peak RSS
  1.79 GB (host-load dependent; not a speedup claim vs the earlier 42 s).
- First delta run (98520e2407, superseded): suite 1 failure + CONTROL_FAIL (facade parity: the width receipt
  string embedded the staged namelist path) and M13 SURVIVED (failed closed with GuardRefused, not an assertion) ->
  fixed in 61ab0781e7; logs kept in `<USER_HOME>/wrf_gpu2_lanes/w41-da-core/delta_98520e2407/`.
- Identity: the delta changes src only under `gpuwrf/assimilation/**`; frozen Swiss/PROD identities are main's
  (not re-measured here; the lane's forecast-path diff vs main is still only the host-only `da check|analyse`
  argv predicate + 2 CLI registration lines).
- New ALISIOS reads (guarded metadata, ALLOW, no V5 case — V5 confirmatory cases are April-June — nothing from
  2026-09-15 on): `namelist.input` of 12 Jan-Feb runs (wg_20260115_18z_a1 .. wg_20260227_18z_a1) for the coverage
  windows, one facade GuardedRead each (receipts `real_e2e/run_0eeb8fff1d/analysis/governance/guard_metadata/`). Seven
  of them belong to archive runs later trimmed by `max_members` (the trim must follow DENY filtering).
- Performance/memory: planning adds one small guarded namelist read per ALISIOS run (12 in the e2e, `da check`
  3.4 s); no forecast-path change.

## What is delivered

| Piece | Where | Notes |
| --- | --- | --- |
| Strict config `gpuwrf.da.v1` | `letkf/config.py` | unknown keys refused; `inflation`/`cycle` sections delegated to the ens/cycle modules; `with_overrides` revalidates |
| Governance plug-ins | `letkf/governance.py`, `letkf/alisios_guard.py` | read plan BEFORE any open; `none` (size+SHA-256 of every input, audit-hook + netCDF4 spy, unplanned opens listed, ALISIOS paths refused), `alisios` (guard scan, archive frame needs ALLOW valid AND issue day, `max_members` trim after DENY filtering, raw ALISIOS obs refused; then PREFERRED backend = ALISIOS reader facade `alisios.guarded_read` sealed from git at 80b076c9: every ALISIOS input scanned + admitted + read once into a private stage inside one `GuardedRead` with READ_SPY, analysis on the stage with every ALISIOS open refused, stage deleted; FALLBACK = `guarded_session` (admit exact set + refusing spy) when the facade is unavailable or a role is read outside its planned path; `options.facade` auto/require/off), `module:factory` |
| Generic obs readers | `letkf/readers.py` | IGRA2 (any station), ISD-lite (+isd-history), CSV surface/profile, ObsBatch JSON; adapters obsdb parquet / MSG mask (ALISIOS-only formats, guarded extracts) |
| Counted moisture policy (R7/R10) | `letkf/moisture.py`, `analysis.py`, `state_io.py` | clip (default) / refuse / log; only analysis-changed cells treated (background negatives kept and counted); dry-air water budget from C3F/C4F/MUB/MU/P_TOP (0.1 kg m-2 per column, 1e-3 of member vapour), float64 budget, non-finite budget refused; written-field positivity check; mean wrfinput gated the same way |
| Hooks | `analysis.analyse` | `maintenance`, `prior_inflation` (per analysis point), `var_loc` groups; defaults bitwise = plain analysis |
| Fixed-QV/MU rebalance | `state_io` / `analysis` (kept) | rebalance ValueError -> inadmissible with reason (no exception) |
| API + CLI | `letkf/api.py`, `letkf/cli.py`, `gpuwrf/cli.py` (+2 lines), `gpuwrf/__init__.py` (offline predicate) | `gpuwrf da check|analyse` backend-dark (no JAX import); `da cycle|members` mounted from the cycle lane and NOT offline (they run `gpuwrf run` in-process) |
| Docs | `docs/da.md` | guide, support row, config reference, evidence with the audited numbers, limits |

No hot-path change: `git diff main..HEAD -- src ':!src/gpuwrf/assimilation'` = `gpuwrf/cli.py` (+2 lines:
`register_da(subparsers)`) and `gpuwrf/__init__.py` (offline predicate, `argv[1:3]` in `da check|analyse`).

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Analytic Kalman + L96 | PASS | `tests/da/test_letkf.py`: transform == state-space Kalman update with ensemble covariance (also inflated), scalar gain for one obs, Lorenz-96 OSSE 3 seeds, default hooks bitwise, per-point inflation constant == scalar bitwise |
| Product tests | **315 passed, 12 skipped** (3.5 min, peak RSS 1.1 GB: slightly above the 1 GB RAM-slot threshold, run without a slot - disclosed; critic: use `with_ram_slot`) | `tests/da` (104, 0 skipped) + `tests/a1_da` on snapshot 987fbe8628; the 12 skips are a1-da real-data fixtures no longer on disk (guarded copies cleaned, scratch 0503 case), not blocked reads: `<USER_HOME>/wrf_gpu2_lanes/w41-da-core/final_987fbe8628/da_tests.log` (+ junit) |
| Mutants (genuine assertion kills, 0 setup errors, E202) | **24/24 KILLED, control PASS** | `run_mutants.py` on snapshot 987fbe8628: `<USER_HOME>/wrf_gpu2_lanes/w41-da-core/final_987fbe8628/MUTANTS_987fbe8628.json` (M01-M04 algebra/hooks, M05 maintenance, M06-M09 + M17-M19 moisture/budget, M10-M11 config, M12-M14 + M20 governance, M15-M16 readers, M21-M24 facade/spy) |
| Real-data CPU e2e, guarded dev case | PASS | see below |
| Default identity | EXACT (carried forward) | Swiss `3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57`, PROD d01 `196aed9f9973499c5610fdf9965223503a61944156b8cd41357fd8c6e911911e`, measured on snapshot 8b5e91f0b4 (`<USER_HOME>/wrf_gpu2_lanes/w41-da-core/identity/{swiss,prod}.json`). 8b5e91f0b4..CANDIDATE differs in src only in `gpuwrf/assimilation/**` and the `__init__` argv predicate, which is False for `gpuwrf run` and for `release_identity.py` in both versions; main merge brought only `scripts/release/**` |
| Mutant for the offline predicate | KILLED | blanket `"da"` entry -> 2 assertion failures (`test_only_host_only_da_subcommands_skip_the_package_bootstrap[cycle|members]`), run by hand (the overlay runner only mutates the letkf package) |
| S1 shared consumers | **336/336 PASS** (0 fail/error/skip, 29:44, largest chunk 9.0 GB RSS) | snapshot 987fbe8628, one 12G RAM-slot ticket, `proofs/w41_da_core/s1/run_s1.py`: the 226 light nodes in one process, the 110 heavy nodes (v013 smoke, v060 dispatch, m7 restart) in fresh processes of 8, sequentially (`proofs/w41_da_core/s1/S1_SUMMARY_987fbe8628.json`). One process does not fit: two 6G runs ended SIGTERM after 23 tests and a 12G run was kernel-OOM-killed at a 12 GB peak after 30 (user journal: scope result oom-kill); compiled programs accumulate per v013 parametrization. On the final merge 03d1e8d4db (main 1b98beb9d3: wdm57 physics firewall + docsite link test) the light set + tests/da + tests/test_docsite_links.py are re-run (**106 + 226 PASS**, <USER_HOME>/wrf_gpu2_lanes/w41-da-core/final_03d1e8d4db/); the heavy set is carried forward (the added code is wdm57's, whose lane ran the same 336) |
| Facade parity | PASS | `tests/da/test_alisios_facade.py` 6/6 (inside the tests/da run) |

### Real-data CPU end to end (guarded)

`proofs/w41_da_core/real_e2e/run_e2e.py`: A1 development case 0227 (control `wg_20260227_18z_a1`
frame 2026-02-28 12Z, template its `wrfinput_d01`), 12 same-phase Jan-Feb archive members (max_age 45 d;
runs 0115..0225, no V5 confirmatory run), innovation-scaled, Güímar IGRA2 sonde 2026-02-28 12Z (guarded
extract, reported 09:15Z), `governance.plugin = alisios`, then guarded `write_cycle_case(hours=12)` +
`gpuwrf run --dry-run` admission.

Run 6559d46308 (`<USER_HOME>/wrf_gpu2_lanes/w41-da-core/real_e2e/run_6559d46308/E2E_RECEIPT.json`):
check/analyse rc 0, admissible; 392 sonde values, 276 upstream QC, 8 non-finite H(x), 108 assimilated;
O-B -> O-A rms: T 1.90 -> 1.24 K, qv 6.7e-4 -> 3.6e-4 kg/kg, u 3.76 -> 2.97, v 3.63 -> 2.60 m/s;
moisture 0 negative cells, 0 added water, written QVAPOR min 6.5e-11, 0 negative; case admission
ensemble_case / lower_boundary / land_cycle_marker / cli_dry_run PASS; peak RSS 1,687 MB.
That run admitted all 33 candidate frames although 12 were opened -> fixed (8c6069a304: trim after the
DENY filter, so admitted == opened). Rerun f6cb4bf582 (fallback): admission 5 issue / 10 valid days, analysis
333/333 variables bitwise equal to run 6559d46308 (only `created_utc` and the land-cycle marker attribute differ).
Rerun 36e0b6e084 **through the facade with the real guard** (`run_36e0b6e084/E2E_RECEIPT.json`): 15 files (control,
template, namelist, 12 frames) with 15 scan / 15 admit / 15 `authorized_attempt` in READ_SPY, 0 refused,
analysis-phase ALISIOS opens 0, stage removed; analysis 333/333 variables bitwise equal to the fallback run; same
O-B/O-A; case admission 4/4 PASS. (That run's wrfinput attribute still named the staged paths; b01a98b1ef records
the original paths, `inputs_staged: true`; covered by the parity test.)

### ALISIOS reader facade (manager 10:28Z)

- Import: `alisios_guard.seal_facade` writes `src/alisios/{__init__,guard_call,guarded_read}.py` from the ALISIOS
  git object store at `FACADE_COMMIT` 80b076c9 (git blob ids verified, a tampered copy is rewritten) to
  `~/.cache/gpuwrf/alisios_sealed/<commit>/` (option `sealed_dir`); imported with `sys.modules['alisios']` saved and
  restored, so a working-tree `alisios` (fallback) is untouched. The guard script stays the live ALISIOS
  `ops/guards/blocklist_guard_v1_10.py` (the guard writes violation files next to itself and reads
  `<DATA_ROOT>/alisios/state`), passed explicitly, SHA-256 recorded.
- Facade protected roots = the targets' run directories: `GuardedRead.__enter__` runs the guard while its audit
  hook is active, and the guard's own metadata under `<DATA_ROOT>/alisios/state` must stay readable. Our refusing
  spy still covers all ALISIOS data roots during staging and refuses every ALISIOS open during the analysis.
- Why staging: inside a `GuardedRead` every `netCDF4.Dataset` must be read-only on an admitted file, so the
  analysis (template copy, wrfinput write, non-ALISIOS inputs) cannot run inside it. Each file is read exactly
  once (one READ_SPY `authorized_attempt` per file). The stage is about 190 MB for the real case (14 files) and
  is deleted in `finally`.
- Parity (`tests/da/test_alisios_facade.py`, synthetic ALISIOS-like tree from the Swiss case, the same fake guard
  in both backends): admitted (issue, valid) union identical; one admit and one READ_SPY `authorized_attempt` per
  staged file; analysis wrfinput all variables bitwise equal and provenance names equal; receipts equal; a DENY
  day is refused before any open in both (READ_SPY counters 0/1/1, no stage left); every original ALISIOS open is
  refused during the staged analysis; sealed copy == git blob, tampered copy rewritten; backend choice/fallback
  reasons; `require` refuses.
- Test-harness trap found on the way: `tests/da/conftest.py` is an initial conftest (loaded before the mutant
  overlay swaps modules), so it held the ORIGINAL `alisios_guard` and its own blocking hid alisios_guard mutants;
  it now looks the module up per call (M24 was the canary).

## Performance and memory

Host-only analysis between forecasts; the forecast program is unchanged (identity above), so no GPU
time, launches or VRAM change. Real e2e (12 members, 120 x 70 x 44 d01, coarse analysis grid 61 x 36 x 44,
108 obs, 4 worker threads, nice 19, shared lane cores): analyse CLI 122 s at 6559d46308 (load 41.6 s, analysis
phases 62.4 s: weights 45.1, localisation 12.1, update 5.0, rebalance 0.25; SMT siblings busy with a 4 GB identity
job) and 42.3 s at 36e0b6e084 through the facade (stage + load 3.7 s, phases 26.8 s: weights 19.5, localisation 4.4,
update 2.3, rebalance 0.21); peak RSS 1.69 / 1.74 GB. The facade adds one read of every ALISIOS input into a
private stage (about 190 MB for 15 files, deleted at the end) and the import of its seven reader libraries. The A1 probe measured 10.9 s of analysis phases for 10 members
on 2 uncontended cores; the difference here is contention and 12 members, not a measured algorithmic
regression [I]. Memory grows with members x analysed fields (float32 frames + float64 work arrays for
QVAPOR). Governance hashing reads every input once (SHA-256); `hash_inputs: false` disables it.

## Audit items not inherited

`docs/da.md` uses only audit-OK numbers: pooled MSE T2 -15 % [-21, -8], Q2 -29 % (two cases), U10 -27 %,
V10 +6 % [-1, +12] (F16/F17, 2-h block bootstrap, descriptive); analysis cost 10.9 s phases / 16.7 s
CLI + mean write (E20 corrected); spread loss 28-76 % per 6 h (S01 corrected). Not claimed: GPU/CPU
member equivalence (E07/E08), seam neutrality (E14), "clip needed" (S09), improved forecasts (F14),
inversion conclusions (I-rows). Counted clip is a changed policy, with its own budget gate.

## ALISIOS reads (all guarded, ALLOW, CALIBRATION_READ, no V5 case/holdout, nothing from 2026-09-15 on)

- IGRA2 SPM00060018 one launch 2026-02-28 12Z (MAILBOX 09:15Z; receipt `real_e2e/sonde_0228_12z/EXTRACT_RECEIPT.json`).
- CPU-WRF d01 frames via the `alisios` plug-in, three runs of the same e2e (6559d46308 fallback, f6cb4bf582
  fallback, 36e0b6e084 facade): control + template + namelist of wg_20260227_18z_a1 and 12 archive frames at 12Z
  (issue days 20260218..20260225 after the trim); receipts under each run's `analysis/governance/`.

## Limits (also in docs/da.md)

d01 only; MU not analysed; shared LBCs; no cloud/radiance assimilation (cloud only withheld); no saturation
adjustment (supersaturation keep only); forecast skill not established;
`da cycle|members` belong to w41-da-cycle (separate critic).

## Support-matrix row (v0.4.1, proposed)

| `gpuwrf da check|analyse` | Experimental (opt-in, CPU, d01) | Offline LETKF analysis: analytic Kalman, Lorenz-96, mutants, real-file e2e whose analysis passes the case preflight; counted moisture policy with water-budget gate; read governance plug-ins. Forecast skill not established; no nests, MU, cloud obs. Guide |

## Critic quick re-run (fits the 90-minute budget)

```bash
cd <snapshot of the candidate>; ln -s <USER_HOME>/src/wrf_pristine data/wrf_pristine
# 122 product tests incl. facade parity (1.5-9 min depending on host load; the parity test needs the ALISIOS repo)
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 scripts/with_ram_slot.sh --label <critic> -- taskset -c <cores> nice -n 19 python -m pytest tests/da -q -p no:cacheprovider --basetemp <dir>
# single mutants (E202: only assertion failures count); M00 = unmutated control over all of tests/da
W41_MUTANT_ONLY=M25,M27,M30,M32,M34,M37,M39 W41_MUTANT_TMP=<dir>   # delta items 1-4; M19,M24 original JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src \
  python proofs/w41_da_core/mutants/run_mutants.py <out.json>
# release identity (about 1.5 + 3 min each, 4 GB RAM -> with_ram_slot)
python proofs/s1_port/release_identity.py . <out>/swiss.json
python proofs/s1_port/release_identity.py . <out>/prod.json <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
```
