#!/usr/bin/env bash
# v45-rc v0.4.5 CPU LAUNCH SMOKE of one GPU entry (no GPU, no GPU lock, no nvidia-smi). Starts the entry exactly like
# the manager sequencer does (setsid bash -c 'echo $$ > pgfile; exec bash <entry>' with GA_START_UTC/GA_END_UTC, group killed
# at cap+1 min) with W45R_LAUNCH_SMOKE=1, pinned to the lane cores (taskset -c 12,28 nice -n 19), and writes the
# LAUNCH_SMOKE receipt (command, redacted child env, model-reach evidence, phase list, measured durations, caps).
# usage: launch_smoke.sh ENTRY_FILE STANDIN_MANIFEST OUT_JSON [ONLY_SLOTS]
# Refuses while /tmp/wrf_gpu2_quiet or /tmp/wrf_gpu2_gpu_window exists, and when <DATA_ROOT> free - smoke forecast (1 GiB)
# < 365 GiB (the same w45_disk_ok code path; the production forecast is evaluated dry and logged).
set -uo pipefail
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/lib.sh
(( $# >= 3 )) || { sed -n 2,8p "$0"; exit 2; }
ENTRY=$1 MAN=$2 OUTJ=$3 ONLY=${4:-}
if [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; then echo "WAIT: QUIET or GPU window"; exit 3; fi
W45R_LOG=${OUTJ%.json}.guard.log
export W45R_LAUNCH_SMOKE=1 W45R_SMOKE_FORECAST_GIB=1 W45R_SMOKE_CAP_FACTOR=4   # factor: [M] CPU arm 2.3x GPU + slot waits 7-9 min   # [I] <= ~0.3 GB per smoke arm (CPU Swiss cache/seal/1 h wrfouts) x margin, 1 GiB = df resolution
w45_disk_ok "$(sed -n 's/^START_FORECAST_GIB=//p' "$ENTRY")" || { echo "WAIT: <DATA_ROOT> free $(w45_free_gib <DATA_ROOT>) GiB - smoke forecast $W45R_SMOKE_FORECAST_GIB < $W45R_DATA_FLOOR_GIB (retry later)"; exit 95; }
grep -q '"standin": true' "$MAN" || { echo "REFUSED: smoke manifests must be standin"; exit 2; }
CAP=$(( $(sed -n 's/^CAP_MIN=//p' "$ENTRY") * 4 ))   # = entry CAP_MIN x W45R_SMOKE_CAP_FACTOR
P0=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/rc_bundle_3e5219bc8088
ROOT=$(dirname "$("$W45R_PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundles"]["prod"]["dir"])' "$MAN")")
[[ $ROOT == $W45R_SCRATCH/standin_* ]] || { echo "REFUSED: standin root $ROOT"; exit 2; }
SR=$ROOT/smoke; mkdir -p "$SR"
QF=$SR/quiet_private; MB=$SR/mailbox_private.md; : > "$MB"
PGF=$SR/pg_$(basename "$ENTRY" .sh)_$$
NOW=$(date -u +%s)
export GA_START_UTC=$(date -u -d @"$NOW" +%FT%TZ) GA_END_UTC=$(date -u -d @"$(( NOW + CAP * 60 + 600 ))" +%FT%TZ)
SMOKE_ENV=(W45R_LAUNCH_SMOKE=1 W45R_SMOKE_FORECAST_GIB="$W45R_SMOKE_FORECAST_GIB" W45R_SMOKE_CAP_FACTOR="$W45R_SMOKE_CAP_FACTOR" W45R_MANIFEST="$MAN" W45R_SMOKE_ROOT="$SR" W45R_SMOKE_DURATION="--hours 1"
  W45R_SMOKE_REPLAY_BUNDLE="$P0" W45R_SMOKE_E41_CACHE="$P0/jax"
  W45R_SMOKE_E41_PROOF="$P0/receipts/A3_wn3_0227_72h/proof/nested_pipeline_run.json"
  W45R_SMOKE_QUIET_FILE="$QF" W45R_SMOKE_MAILBOX="$MB" W45R_SMOKE_CORES=12,28 W45R_SMOKE_ONLY_SLOTS="$ONLY")
T0=$(date +%s.%N)
env "${SMOKE_ENV[@]}" setsid bash -c "echo \$\$ > '$PGF'; exec taskset -c 12,28 nice -n 19 bash '$ENTRY'" > "$SR/$(basename "$ENTRY" .sh).out" 2>&1 &
pid=$!; sleep 1; PG=$(cat "$PGF" 2>/dev/null || echo "$pid")
deadline=$(( $(date +%s) + CAP * 60 + 60 )); RC=
while kill -0 "$pid" 2>/dev/null; do
  if (( $(date +%s) > deadline )); then kill -TERM -- -"$PG" 2>/dev/null; sleep 20; kill -KILL -- -"$PG" 2>/dev/null; RC=124; break; fi
  sleep 5
done
[[ -z $RC ]] && { wait "$pid"; RC=$?; }
T1=$(date +%s.%N)
sleep 2
OUT=$SR/$(basename "$ENTRY" .sh).out
RUN=$(sed -n 's/^ENTRY .* run \(\S*\) budget.*/\1/p' "$OUT" | tail -1)
# leftovers: the sequencer's group/session AND everything naming the run dir (the supervisor's own setsid session)
LEFT=$(ps -eo pid=,pgid=,sid=,args= | awk -v g="$PG" -v r="${RUN:-/nonexistent-run}" '($2==g || $3==g || index($0, r)) && !/ awk / && !/shell-snapshots/' | head -20)
"$W45R_PY" -I - "$OUTJ" "$ENTRY" "$MAN" "$RC" "$T0" "$T1" "$OUT" "${RUN:-}" "$LEFT" "$PG" "$ONLY" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
outj, entry, man, rc, t0, t1, out, run, left, pg, only = sys.argv[1:12]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return None
def txt(p):
    try: return Path(p).read_text(errors="replace")
    except OSError: return ""
R = Path(run) if run else None
arms = {}
if R and (R / "runs").is_dir():
    for d in sorted(p for p in (R / "runs").iterdir() if p.is_dir()):
        rec = load(d / "RECEIPT.json") or {}
        arms[d.name] = dict(rc=txt(d / "rc.txt").strip() or None, device=txt(d / "device.txt").strip()[:80],
            command=txt(d / "command.txt").strip(), timer=load(d / "timer.json"),
            proof_verdict=((rec.get("proof") or {}).get("verdict")), all_domains_finite=((rec.get("proof") or {}).get("all_domains_finite")),
            n_wrfout=rec.get("n_wrfout"), max_rss_gib=rec.get("max_rss_gib"),
            consumer=[l for l in txt(R / "runs" / f"{d.name}.scope.log").splitlines() if l.startswith("BUNDLE_CONSUMER")][-1:],
            dry_run_plan=("dry_run" in txt(d / "run.log") or '"effective' in txt(d / "run.log")),
            run_log_tail=txt(d / "run.log").splitlines()[-3:], env_child=txt(d / "env_child.txt").splitlines(),
            env_product_platform=[l for l in txt(d / "env_product.txt").splitlines() if l.startswith(("JAX_PLATFORMS", "CUDA_VISIBLE"))])
sess = txt(R / "session.log") if R else ""
phases = [l[21:] for l in sess.splitlines() if re.search(r" (START|PRIME|SEAL|REPLAY|ARM|SKIP|SMOKE|check after|final check|DONE)", l)]
res = dict(schema="v45-rc.launch-smoke.v1", entry=entry, entry_sha256=sha(entry), manifest=man, manifest_sha256=sha(man),
    launcher_command=f"setsid bash -c 'echo $$ > PGFILE; exec taskset -c 12,28 nice -n 19 bash {entry}' (env W45R_LAUNCH_SMOKE=1 + W45R_SMOKE_*, GA window now..now+cap+10 min)",
    rc=int(rc), wall_s=round(float(t1) - float(t0), 1), pgid=int(pg), processes_left_in_group=left.strip().splitlines(),
    run_dir=run, only_slots=only or None, entry_json=load(R / "ENTRY.json") if R else None,
    manifest_check=(load(R / "MANIFEST_CHECK.json") or {}).get("checks") if R else None,
    lifecycle=load(R / "LIFECYCLE.json") if R else None, entry_result=load(R / "ENTRY_RESULT.json") if R else None,
    arms=arms, session_phases=phases, entry_log=txt(R / "entry.log").splitlines() if R else [],
    post_log=txt(R / "post.log").splitlines() if R else [], e41=load(R / "e41/E41_POST.json") if R else None,
    disk_guard=[l for l in (txt(outj[:-5] + ".guard.log") + txt(out) + sess).splitlines() if "DISK-GUARD" in l],
    stdout_tail=txt(out).splitlines()[-15:])
Path(outj).write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps(dict(rc=res["rc"], wall_s=res["wall_s"], left=len(res["processes_left_in_group"]), arms={k: (v["rc"], v["device"][-12:], v["proof_verdict"]) for k, v in arms.items()}), indent=1))
PY
exit "$RC"
