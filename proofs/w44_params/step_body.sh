#!/usr/bin/env bash
# V3 (2026-10-24, manager 22:37Z): the FIRST production entry (diffusion prepare off d01) is the lane's GPU smoke.
# Every other entry refuses before any GPU/CPU model work unless that entry left a PASS receipt for THIS tree.
# CPU stages run in their own MemoryMax/no-swap scopes (no shared RAM-slot ticket inside a GPU window, 15:09Z rule).
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
LANE=<USER_HOME>/wrf_gpu2_lanes/w44-params
CASE=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
family=${1:?family}; stage=${2:?stage}; shift 2
smoke=0; for arg in "$@"; do [[ $arg != --cpu-smoke ]] || smoke=1; done
[[ ${PARAMS_ARM_SUPERVISOR_PID:-} == "$PPID" ]] || { echo 'whole-arm supervisor required'; exit 9; }
[[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
HEAD=$(git -C "$TREE" rev-parse HEAD)
ROOT=$LANE/gpu_v3; [[ $smoke != 1 ]] || ROOT=$LANE/launch_smoke_v3
RUN=$ROOT/$family; CACHE=$ROOT/cache; FIRST=$ROOT/FIRST_ENTRY_PASS.json
entry="$family-$stage${1:+-$1}${2:+-$2}"; [[ $stage == prepare ]] || entry="$family-$stage"
source "$HERE/gpu_environment.sh"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu mkdir -p "$RUN/tmp" "$CACHE/jax" "$CACHE/autotune" "$CACHE/triton" "$CACHE/cuda"
stamp=$(JAX_PLATFORMS=cpu date -u +%Y%m%dT%H%M%S%N)

# Own memory scope for CPU stages; refuse rather than squeeze the ALISIOS/OS reserve. Lane cores (AGENTS §4);
# PARAMS_CPU_CORES narrows them when the manager lends cores to another job (e.g. 22:49Z: 24,25,28 only).
scope() {
  local max=$1; shift
  local avail; avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  (( avail >= 32 )) || { echo "MemAvailable ${avail} GiB < 32 GiB: CPU stage refused"; return 8; }
  local core; for core in ${PARAMS_CPU_CORES//,/ }; do
    [[ " 8 9 12 13 24 25 28 29 " == *" $core "* ]] || { echo "core $core is not a lane core"; return 8; }; done
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; return 3; fi
  JAX_PLATFORMS=cpu systemd-run --user --scope --quiet -p MemoryMax="$max" -p MemorySwapMax=0 -- \
    taskset -c "${PARAMS_CPU_CORES:-8,9,12,24,25,28}" nice -n 19 "$@"
}

# First-entry gate (both modes, so the CPU rehearsal proves it).
if [[ $entry != diffusion-prepare-off-d01 ]]; then
  [[ -f $FIRST ]] || { echo "first production entry diffusion-prepare-off-d01 has no PASS receipt in $ROOT: $entry skipped"; exit 7; }
  JAX_PLATFORMS=cpu python - "$FIRST" "$HEAD" "$smoke" "$HERE/validate_step.py" <<'PY' || { echo "first-entry PASS receipt invalid: $entry skipped"; exit 7; }
import hashlib, json, subprocess, sys
gate, head, smoke, validator = sys.argv[1:]
g = json.load(open(gate))
assert g["entry"] == "diffusion-prepare-off-d01" and g["tree_head"] == head and g["smoke"] == int(smoke), g
assert hashlib.sha256(open(g["receipt"], "rb").read()).hexdigest() == g["receipt_sha256"], "first receipt changed"
subprocess.run([sys.executable, validator, g["receipt"], smoke, "prepare"], check=True)
PY
fi

child() {
  local seconds=$1 label=$2; shift 2
  if [[ $smoke == 1 ]]; then
    scope 12G timeout --signal=TERM --kill-after=10 300 python "$HERE/supervise.py" --whole-arm --seconds 280 -- \
      env JAX_PLATFORMS=cuda "$@" --cpu-smoke > "$RUN/$label-smoke.log" 2>&1
  else
    local end now
    end=$(date -u -d "${GA_END_UTC:?}" +%s)
    now=$(JAX_PLATFORMS=cpu <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh)
    (( end-now >= 30 )) || return 5
    (( seconds <= end-now-10 )) || seconds=$((end-now-10))
    "$TREE/scripts/with_gpu_lock.sh" --label w44-params --timeout 1 -- \
      timeout --signal=TERM --kill-after=10 $((seconds+5)) python "$HERE/supervise.py" --seconds "$seconds" \
      --deadline-epoch "$end" -- "$@" > "$RUN/$label.log" 2>&1
  fi
}

if [[ $stage == prepare ]]; then
  variant=${1:?off/on}; domain=${2:?domain}
  [[ $variant == off || $variant == on ]]; [[ $domain == d01 || $domain == d02 ]]
  out=$RUN/$variant-$domain-$stamp
  # Prior paid OFF root: ~249s backend/JAX compile +151s export interval.
  # 900s bounds ONE domain/value, not the former aggregate OFF+ON arm.
  child 900 "$variant-$domain-$stamp" python "$HERE/cold_prepare.py" \
    --input-dir "$CASE" --out "$out" --family "$family" --variant "$variant" --domain "$domain"
  JAX_PLATFORMS=cpu python "$HERE/validate_step.py" "$out/receipt.json" "$smoke" prepare
  if [[ $entry == diffusion-prepare-off-d01 ]]; then
    JAX_PLATFORMS=cpu python - "$FIRST" "$out/receipt.json" "$HEAD" "$smoke" <<'PY'
import datetime, hashlib, json, os, sys
gate, receipt, head, smoke = sys.argv[1:]
body = dict(entry="diffusion-prepare-off-d01", receipt=receipt, tree_head=head, smoke=int(smoke),
            receipt_sha256=hashlib.sha256(open(receipt, "rb").read()).hexdigest(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
fd = os.open(gate + ".tmp", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(fd, "w") as handle:
    json.dump(body, handle, indent=1)
os.replace(gate + ".tmp", gate)
print("FIRST ENTRY PASS", gate)
PY
  fi
  exit
fi
[[ $stage == warm ]]
# E41 verdicts (0 PASS, 2 FAIL/INCOMPLETE -> wave_gate.json) are RECORDED and the measurement continues; any other
# exit (crash, import/env error, timeout) means the phase itself is broken and stops the entry (E220).
e41() {
  local rc=0; mkdir -p "$RUN/e41"
  scope 12G env GPUWRF_JAX_CACHE=0 timeout 600 python "$TREE/scripts/wave_cubin_gate.py" "$@" || rc=$?
  echo "$rc" > "$RUN/e41/rc"
  if [[ $rc == 0 || $rc == 2 ]] && [[ -s $RUN/e41/wave_gate.json ]]; then
    echo "E41 phase completed rc=$rc (verdict recorded in $RUN/e41/wave_gate.json)"; return 0
  fi
  echo "E41 phase broken rc=$rc"; return 1
}
if [[ $smoke != 1 ]]; then
  scope 8G timeout 90 python "$HERE/check_prepared.py" "$RUN" "$CACHE" "$family"
  e41 "$CACHE/jax" "$RUN/e41" 'jit__advance_chunk*' --run-proof "$RUN/prepared.json"
else
  # Same builder/env/scope for both CPU phases (E220). A CPU cache has no CUDA ELF, so the rehearsal's E41 scan
  # reads a PAID real GPU donor (v0.4.3 release bundle, read-only); the GPU arm scans its own cache above.
  scope 8G timeout 90 python "$HERE/check_prepared.py" "$RUN" "$CACHE" "$family" --cpu-smoke
  e41 <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax "$RUN/e41" 'jit__advance_chunk*'
fi
out=$RUN/warm-$stamp
child 300 "measure-$stamp" python "$HERE/root_gpu.py" --input-dir "$CASE" \
  --out "$out" --family "$family" --phase measure
if [[ $smoke == 1 ]]; then
  JAX_PLATFORMS=cpu python "$HERE/validate_step.py" "$out/measure_cpu_smoke.json" 1 warm
  child 300 "profile-$stamp" python "$HERE/root_gpu.py" --input-dir "$CASE" \
    --out "$out" --family "$family" --phase measure --profile --reps 3
  JAX_PLATFORMS=cpu python "$HERE/validate_step.py" "$out/measure_profile_cpu_smoke.json" 1 warm
  exit
fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu mkdir -p "$out/profile"
child 300 "profile-$stamp" nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o "$out/profile/roots" \
  python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$out" --family "$family" --phase measure --profile --reps 3
scope 8G timeout 120 nsys export --type sqlite --output "$out/profile/roots.sqlite" "$out/profile/roots.nsys-rep"
scope 4G timeout 60 python "$HERE/profile_metrics.py" "$out/profile/roots.sqlite" "$out/profile/metrics.json"
scope 4G timeout 60 python "$HERE/program_metrics.py" "$out/measure.json" "$RUN/e41" "$out/program_metrics.json"
