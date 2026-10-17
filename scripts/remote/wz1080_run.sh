#!/usr/bin/env bash
# wz1080_run.sh — run a command on the wz1080 test box (GTX 1080 Ti, sm_61) inside a synced tree.
#
# Usage: scripts/remote/wz1080_run.sh [--gpu|--cpu] [--label L] [--timeout S] [--lock-wait S] [--tree SHA|main|PATH]
#                                     [--env K=V]... [--bg] -- <command> [args...]
#        scripts/remote/wz1080_run.sh --status | --tail LABEL | --kill LABEL
#   --cpu (default)  JAX_PLATFORMS=cpu, CUDA hidden, niced (E50: nothing touches the GPU outside the lock)
#   --gpu            under the box's GPU lock (wz-gpu-lock, same proof the product preflight checks)
#   --tree           snapshot <sha> from wz1080_sync.sh (default: the box's main clone); PYTHONPATH=<tree>/src
#   --timeout        hard wall limit of the command (default 3600 s; lock wait not included)
#   --no-nice        CPU mode without nice (timed CPU baselines only)
#   --bg             detach; log under ~/wrf_gpu2_runs/<label>/ on the box; see --status/--tail/--kill
# Pascal notes: release fast paths are Pallas/Triton (Ampere+ only in XLA) -> use --env GPUWRF_FAST_DEFAULTS=0
# for GPU runs unless you are probing exactly that; see scripts/remote/WZ1080_RUNBOOK.md.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=wz1080_common.sh
source "$HERE/wz1080_common.sh"

MODE=cpu; LABEL="${GPUWRF_LANE:-wz}"; TIMEOUT=3600; WAIT=14400; TREE=main; BG=0; ENVS=(); NICE="nice -n 10"
case "${1:-}" in
  --status)
    wz_ssh 'echo "lock: $(cat /tmp/wrf_gpu2_gpu.lock.holder 2>/dev/null)"; echo "queue: $(ls -1 /tmp/wz_gpu_queue 2>/dev/null | sort | sed "s/^[0-9]*_//" | tr "\n" " ")"; for p in wrf_gpu2_runs/*/*.pid; do [ -e "$p" ] || continue;
      pid=$(cat "$p"); if kill -0 "$pid" 2>/dev/null; then echo "RUNNING $p pid=$pid"; fi; done;
      nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null'; exit 0;;
  --tail)
    wz_ssh "f=\$(ls -1t $WZ_R_RUNS/$2/*.log 2>/dev/null | head -1); [ -n \"\$f\" ] && { echo \"== \$f\"; tail -n ${WZ_TAIL:-60} \"\$f\"; }"; exit 0;;
  --kill)
    wz_ssh "p=\$(ls -1t $WZ_R_RUNS/$2/*.pid 2>/dev/null | head -1); [ -n \"\$p\" ] || exit 1; pid=\$(cat \"\$p\");
      kill -TERM -- -\$pid 2>/dev/null; for i in \$(seq 30); do kill -0 \$pid 2>/dev/null || break; sleep 1; done;
      kill -KILL -- -\$pid 2>/dev/null; echo \"killed group \$pid\""; exit 0;;
esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) MODE=gpu; shift;;
    --cpu) MODE=cpu; shift;;
    --label) LABEL="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --lock-wait) WAIT="$2"; shift 2;;
    --tree) TREE="$2"; shift 2;;
    --env) ENVS+=("$2"); shift 2;;
    --bg) BG=1; shift;;
    --no-nice) NICE=""; shift;;
    --) shift; break;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown option $1 (missing '--' before the command?)" >&2; exit 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "no command after '--'" >&2; exit 2; }
[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "label must match [A-Za-z0-9._-]+" >&2; exit 2; }

case "$TREE" in
  main) RTREE="\$HOME/$WZ_R_REPO";;
  /*|~*) RTREE="$TREE";;
  *) [[ "$TREE" =~ ^[0-9a-f]{7,40}$ ]] || { echo "--tree: sha, main or absolute path" >&2; exit 2; }
     RTREE="\$HOME/$WZ_R_WT/${TREE:0:12}";;
esac

CMD_Q="$(printf '%q ' "$@")"
ENV_Q=""
for kv in "${ENVS[@]}"; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "--env needs K=V, got $kv" >&2; exit 2; }
  ENV_Q+="export $(printf '%q' "$kv"); "
done

if [[ "$MODE" == gpu ]]; then
  WRAP="unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES; $ENV_Q exec \$HOME/.local/bin/wz-gpu-lock --label $LABEL --lock-wait $WAIT -- timeout -k 60 $TIMEOUT $CMD_Q"
else
  WRAP="export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false; $ENV_Q exec \${WZ_CPU_CPUS:+taskset -c \$WZ_CPU_CPUS} $NICE timeout -k 60 $TIMEOUT $CMD_Q"
fi
SCRIPT="set -eo pipefail; T=$RTREE; [ -d \"\$T\" ] || { echo \"no tree \$T on the box (run wz1080_sync.sh first)\" >&2; exit 3; }
cd \"\$T\"; source \"\$HOME/$WZ_R_ENV/activate.sh\" \"\$T\"; export GPUWRF_LANE=$LABEL WZ_RUN_DIR=\"\$HOME/$WZ_R_RUNS/$LABEL\"; mkdir -p \"\$WZ_RUN_DIR\"
echo \"[wz1080_run] host=\$(hostname) tree=\$T head=\$(git rev-parse --short=12 HEAD 2>/dev/null) mode=$MODE label=$LABEL\" >&2
$WRAP"

if (( BG )); then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  # E131: record the job's own PID from inside the new session (setsid may fork; $! can be a short-lived parent).
  BGS="d=\$HOME/$WZ_R_RUNS/$LABEL; mkdir -p \$d; printf '%s\n' $(printf '%q' "$*") > \$d/$STAMP.cmd
setsid nohup bash -c 'echo \$\$ > \"\$1\"; exec bash -c \"\$2\"' wzjob \$d/$STAMP.pid $(printf '%q' "$SCRIPT") > \$d/$STAMP.log 2>&1 < /dev/null &
for i in 1 2 3 4 5 6 7 8 9 10; do [ -s \$d/$STAMP.pid ] && break; sleep 0.5; done
echo \"[wz1080_run] started pid=\$(cat \$d/$STAMP.pid) log=\$d/$STAMP.log\""
  wz_ssh "bash -c $(printf '%q' "$BGS")"
else
  ssh -n "${WZ_SSH_OPTS[@]}" "$WZ_HOST" "bash -c $(printf '%q' "$SCRIPT")"
fi
