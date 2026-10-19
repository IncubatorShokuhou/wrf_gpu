"""w42-bundle042: write the RW3T pre-registration PREREG.json into the FROZEN tools dir before the GPU session.

usage: python3 prereg.py FROZEN_TOOLS_DIR
Records the tool SHA-256s, case, arms, abort/acceptance/verdict rules (v0.4.2 common A/B rule; verdict rule unchanged from
the v0.4.1 addendum w41-rw3add PREREG/PREREG2) and the session env. Refuses to overwrite an existing PREREG.json.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

F = Path(sys.argv[1])
out = F / "PREREG.json"
assert not out.exists(), "PREREG.json exists: pre-registration is immutable"
pin = dict(l.split("=", 1) for l in (F / "pin.env").read_text().split() if "=" in l)
tools = sorted(p.name for p in F.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "PREREG.json"))
rec = {
    "schema": "w42-bundle042.rw3t.prereg.v1",
    "registered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "authority": "contract .agent/kernel/lanes/w42-bundle042.md (RW3 'timing per the A/B rules or disclosed'); "
                 "_WAVE_V042_COMMON.md A/B rule (interleaved, one lock session, core watch 3,13,19,29, sweep-class "
                 "abort, everything else disclosed with the arm's own runqueue wait + cgroup CPU pressure, A/A and B/B "
                 "spreads); verdict rule = w41-rw3add PREREG c423594b / PREREG2 b8e1fef2 (unchanged)",
    "pin": pin,
    "tools_frozen_dir": str(F),
    "tools_sha256": {n: hashlib.sha256((F / n).read_bytes()).hexdigest() for n in tools},
    "case": {"name": "WN3 0227 3-nest (wg_20260227_18z_a1)", "dir": "<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/cases/wn3_0227",
             "provenance": "copy of the w42-harness guarded facade_stage copy (STAGED_INDEX 24/24 SHA match; "
                           "<USER_HOME>/wrf_gpu2_lanes/w42-harness/stage_receipt/facade_stage.json)",
             "cli_args": "--domains-from-namelist --emit-initial-history --hours H"},
    "arms": {"N": "v0.4.2 sealed bundle <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2 (its tools/bundle_consumer.sh, fresh layer)",
             "O": "v0.4.1 sealed bundle <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1 (its tools/bundle_consumer.sh, fresh layer; "
                  "READ-ONLY, BUNDLE_OK before/after)",
             "order": "N72 N1 O1 N2 O2 N3 O3 O72 (N72 first: it is also the RW1 72 h warm replay and the RW3 D6 input)",
             "session": "phase 2 of ONE with_gpu_lock session (phase 1 = prime/seal/replays); window end = min(session "
                        "start + 120 min, 2026-10-19T18:30:00Z); a slot starts only if its need (24 h 270 s, 72 h 650 s) fits"},
    "abort_rules": {"watcher": "core_watch.py (frozen a4938233) 1 s on cores 3,13,19,29, own session tree excluded",
                    "rule_P": "one foreign process > 100 % of one core on the gated cores for > 10 s cumulative -> abort",
                    "rule_A": "sum of foreign on-core shares > 200 % for > 10 s cumulative -> abort",
                    "shadow": "never-aborting watcher at the 20/50 % contract rules -> disclosure + strict subset",
                    "repeat": "aborted arm kept as <slot>.abortedK, slot repeated with fresh dir + layer, max 3 tries",
                    "pre_gate": "60 s quiet stretch (same thresholds at 3 s), rolling, max 180 s per slot, then start "
                                "(BUSY_START disclosed); busy while updatedb runs",
                    "hold": "manager HOLD/STOP line to w42-bundle042 stops before the next arm",
                    "post_arm_contention": "none gated (v0.4.2 common rule): CLI main-thread runqueue wait share, cgroup "
                                           "cpu.pressure, <DATA_ROOT> await, foreign agg mean are DISCLOSED per arm"},
    "bundle_checks": "compile_bundle.py check BUNDLE_OK missing=0 changed=0 new=0 on the arm's bundle before AND after every "
                     "arm, else stop the session",
    "acceptance": "accept.py: rc 0 + CUDA device; BUNDLE_CONSUMER rc=0 compiled_new=0 + both checks OK; watcher present/"
                  "healthy; final within_rule, no ABORT; continuous coverage t0..t_end",
    "metrics": {"primary": "stepping median per arm = median of the d03 hourly publication intervals, leads 2..H "
                           "(s per forecast hour, all 3 domains; attrib_detail d03_step_median) - manager 2026-10-19T05:59:13Z",
                "secondary_disclosed": ["whole-run s/fc-h = last original wrfout mtime minus process t0, / H (E162); an arm "
                                        "whose arm-window <DATA_ROOT> (nvme3n1) await > 50 ms (disk_watch; iostat -x logged "
                                        "per arm) is whole-run DISTURBED: disclosed, excluded from the whole-run pooled "
                                        "ratio, never called a regression",
                                        "forecast-only wall (receipt)", "d03 cadence mean", "d01 cadence median"]},
    "verdict_rule": {"basis": "counted 24 h arms; pooled N/O = mean(N) / mean(O) of the primary metric; spread S = max(A/A "
                              "over O, B/B over N) with A/A = max/min - 1 over counted arms of one kind",
                     "PASS": "pooled N/O + S <= 1.10", "REAL_REGRESSION": "pooled N/O - S > 1.10 -> STOP, name suspects",
                     "INCONCLUSIVE": "otherwise, or fewer than 2 counted arms of either kind",
                     "pairs": "per-pair N_i/O_i reported", "72h": "N72/O72 reported as one pair (not in the pooled rule)",
                     "strict": "arms whose shadow watcher would also pass 20/50 % reported separately"},
    "budgets": {"24h": "arm.sh 330 s, scope RuntimeMaxSec 420 s, MemoryMax 20G", "72h": "arm.sh 760 s, RuntimeMaxSec 860 s"},
    "env": "AT_* defaults of rw3t.sh (RULE_P 100, RULE_A 200, QGATE 60 s, GATE_MAX 180 s, WATCH_CPU 8, SAMPLE_CPU 9) unless "
           "the session log START line says otherwise",
}
out.write_text(json.dumps(rec, indent=1) + "\n")
print("PREREG", hashlib.sha256(out.read_bytes()).hexdigest())
