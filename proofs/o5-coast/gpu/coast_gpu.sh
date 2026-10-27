#!/usr/bin/env bash
# o5-coast GPU launcher: root ring-0 WRF walk (GPUWRF_ROOT_SPEC_STAGE_CLOCK=walk, v0.4.4 default candidate) from an IMMUTABLE
# git-archive export of lane commit $SRC_SHA, private caches, raw history streamed through stream_extract.py.
# Production case: WN3 20260227_18z 3-nest (our own CPU twin inputs, SHAs == teacher 0227); reference = our CPU twin.
#   MINI : walk, 3 h (pays the cold compile into the private cache)            WALK : walk, 72 h (warm on MINI's cache)
#   SEED : walk, 72 h on case0227_3nest_seed (o5-9nest perturb_theta.py, +-1 ulp total theta) -> GPU self-twin G
#   REL  : GPUWRF_ROOT_SPEC_STAGE_CLOCK=0 (released clock, cold)   ON1/ON2 : stage-end / stage-midpoint pins (diagnostics)
# --mini (CUDA mini, V044 GPU-window rule): SAME builder/env/phases, case = Swiss d01 example (our own, RD11 inputs),
#   1 h, arm MINI only, cap 10 min; checker also reports the d01 MU ring offset vs our Swiss CPU-WRF at +1 h.
# Phases (every run): checks -> GPU lock (own with_gpu_lock, sequencer-compatible) -> arms (model + stream extract) ->
#   checker (CPU, seconds) -> E41 cubin gate (CPU, detached; waits while /tmp/wrf_gpu2_gpu_window exists, RAM slot 8G).
# usage: GA_START_UTC=.. GA_END_UTC=.. bash coast_gpu.sh [--mini] [ARMS]        (ARMS default MINI,WALK; --mini: MINI)
#        bash coast_gpu.sh --dry-run [--mini] [ARMS]   CPU launch smoke of EVERY phase (no GPU)
set -uo pipefail
SRC_SHA=__SRC_SHA__
L=<DATA_ROOT>/wrf_gpu2/scratch/o5-coast
EXP=$L/src_$SRC_SHA
TOOLS=$L/tools___LAUNCH_SHA__/proofs/o5-coast/tools   # read-only export of the launcher commit (tools evolve, src fixed)
MAILBOX=${GA_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
CPUS="taskset -c 8,9,12,24,25,28 nice -n 19"
MODE=outer; MINI=0
while [[ ${1:-} == --* ]]; do case $1 in --dry-run) MODE=dry;; --locked) MODE=locked;; --mini) MINI=1;; *) echo "bad option $1"; exit 2;; esac; shift; done
if (( MINI )); then
  ARMS=${1:-MINI}; CASE=$L/case_swiss; CPUREF=<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu; NDOM=1; CAP_ARM_S=${GA_CAP_ARM_S:-600}
  CASE_SHA=$L/case_swiss.sha256; INPUTS="namelist.input wrfinput_d01 wrfbdy_d01"; MINI_HOURS=1; TAG=mini
  EIH=""   # native single-domain d01 writes the t0 frame itself; the CLI refuses --emit-initial-history there
else
  ARMS=${1:-MINI,WALK}; CASE=$L/case0227_3nest; CPUREF=<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu; NDOM=3
  CAP_ARM_S=${GA_CAP_ARM_S:-2700}; CASE_SHA=$L/case0227_3nest.sha256; MINI_HOURS=3; TAG=prod; EIH="--emit-initial-history"
  INPUTS="namelist.input wrfinput_d01 wrfinput_d02 wrfinput_d03 wrfbdy_d01 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03"
fi
SEEDCASE=$L/case0227_3nest_seed
rnow() { local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] || { echo "real_utc.sh non-numeric" >&2; exit 7; }; echo "$e"; }
post() { [[ $MODE == dry ]] && { echo "POST(dry): $*"; return; }; printf '%s o5-coast: [real %s] %s\n' "$(date -u +%FT%TZ)" "$("$REAL" --iso)" "$*" >> "$MAILBOX"; }
die() { echo "REFUSED: $*"; [[ $MODE == dry ]] || post "GPU NOT STARTED ($TAG): $*"; exit "${2:-2}"; }
# ---- static checks (every mode) ----
[[ $ARMS =~ ^(MINI|WALK|SEED|REL|ON1|ON2)(,(MINI|WALK|SEED|REL|ON1|ON2)){0,5}$ ]] || die "arms must be a comma list of MINI,WALK,SEED,REL,ON1,ON2"
(( MINI )) && [[ $ARMS != MINI ]] && die "--mini runs arm MINI only"
[ -d "$EXP/src/gpuwrf" ] || die "missing export $EXP"
[ -s "$TOOLS/stream_extract.py" ] && [ -s "$TOOLS/check_arm.py" ] || die "missing tools export $TOOLS"
[ "$(cat "$EXP/EXPORT_COMMIT")" == "$SRC_SHA" ] || die "export commit mismatch"
( cd "$EXP" && sha256sum -c --quiet EXPORT_SHA256SUMS ) > /dev/null 2>&1 || die "export content changed (EXPORT_SHA256SUMS)"
[ -L "$EXP/data/wrf_pristine" ] && [ -d "$EXP/data/wrf_pristine/WRF" ] || die "data/wrf_pristine link missing in export"
for f in $INPUTS; do
  [ -s "$CASE/$f" ] || die "missing $CASE/$f"
  case "$(readlink -f "$CASE/$f")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/*) die "ALISIOS-root input $f";; esac
done
( cd "$CASE" && sha256sum -c --quiet "$CASE_SHA" ) > /dev/null 2>&1 || die "case inputs != recorded sha256 ($CASE_SHA)"
if [[ $ARMS == *SEED* ]]; then
  python3 - "$SEEDCASE" "$CASE_SHA" <<'PY' || die "seed case check failed"
import hashlib, json, sys
from pathlib import Path
d, ref = Path(sys.argv[1]), dict(reversed(l.split()) for l in open(sys.argv[2]) if l.strip())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = json.load(open(d / "PERTURB.json"))["domains"]
for name, want in ref.items():
    if name.startswith("wrfinput_"):
        r = rec[name[-3:]]
        assert r["src_sha256"] == want and sha(d / name) == r["dst_sha256"], name
    else:
        assert sha(d / name) == want, name
print("seed case OK")
PY
fi
FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1)
(( FREE_G >= 370 )) || die "<DATA_ROOT> free ${FREE_G} GiB < 370" 4
hours_of() { [[ $1 == MINI ]] && echo "$MINI_HOURS" || echo 72; }
checker() {  # $1 arm dir, $2 arm
  JAX_PLATFORMS=cpu $CPUS timeout 300 python "$TOOLS/check_arm.py" "$1" "$(( NDOM * ($(hours_of "$2") + 1) ))" "$CPUREF"; }
e41() {  # $1 cache/jax, $2 out dir, $3 run proof
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$EXP/src" $CPUS timeout 900 python "$EXP/scripts/wave_cubin_gate.py" "$1" "$2" \
    ${3:+--run-proof "$3"}; }
# ---- dry: CPU launch smoke of every phase ----
if [[ $MODE == dry ]]; then
  STAMP=${GA_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}; OUT=$L/dry_${TAG}_$STAMP; mkdir -p "$OUT"
  bash -n "$0" || die "syntax"
  for arm in ${ARMS//,/ }; do
    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$EXP/src" $CPUS timeout 300 python -m gpuwrf.cli run --input-dir \
      "$( [[ $arm == SEED ]] && echo "$SEEDCASE" || echo "$CASE")" --output-dir "$OUT/$arm/wrfout" --domains-from-namelist \
      $EIH --hours "$(hours_of "$arm")" --dry-run > "$OUT/cli_dry_$arm.json" 2> "$OUT/cli_dry_$arm.err"
    echo "phase model($arm) cli --dry-run rc=$?"
  done
  JAX_PLATFORMS=cpu $CPUS timeout 60 python -c "import sys; sys.path.insert(0,'$TOOLS'); import stream_extract, load_raw" \
    || die "extractor import failed"; echo "phase extract import OK"
  mkdir -p "$OUT/MINI/frames" && : > "$OUT/MINI/frames/MANIFEST.jsonl"
  checker "$OUT/MINI" MINI > "$OUT/checker.txt" 2>&1; echo "phase checker started rc=$? (expected 1 = FAIL on the empty dry arm): $(tail -1 "$OUT/checker.txt")"
  mkdir -p "$OUT/empty_cache"; e41 "$OUT/empty_cache" "$OUT/E41" "" > "$OUT/e41.txt" 2>&1; r=$?
  grep -q "No module named" "$OUT/e41.txt" && die "E41 import failure (env)"
  echo "phase E41 started rc=$r (empty cache -> refusal expected): $(tail -1 "$OUT/e41.txt" | cut -c1-160)"
  echo "DRY OK: $TAG export $SRC_SHA, case sha OK, free ${FREE_G} GiB, arms $ARMS, receipts $OUT"
  exit 0
fi
# ---- outer: lock + checker + detached E41 (sequencer entry) ----
if [[ $MODE == outer ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  export GA_STAMP=${GA_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}; OUT=$L/gpu_${TAG}_$GA_STAMP
  [ -e "$OUT" ] && die "output root exists $OUT"
  rc=0
  "$EXP/scripts/with_gpu_lock.sh" --label o5-coast --timeout 60 -- bash "$0" --locked $( (( MINI )) && echo --mini ) "$ARMS" || rc=$?
  for a in ${ARMS//,/ }; do [ -d "$OUT/$a" ] && { checker "$OUT/$a" "$a" > "$OUT/$a/checker.txt" 2>&1; echo "checker $a rc=$?" >> "$OUT/rc_checker.txt"; }; done
  proof=$(ls "$OUT"/*/wrfout/proofs/nested_pipeline_run.json 2>/dev/null | head -1)
  if [ -d "$OUT/cache/jax" ]; then
    setsid nohup bash -c "while [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 60; done; \
      <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast/scripts/with_ram_slot.sh --label o5-coast-e41 --max 8G --timeout 7200 -- \
      env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH='$EXP/src' $CPUS timeout 900 python '$EXP/scripts/wave_cubin_gate.py' \
      '$OUT/cache/jax' '$OUT/E41' ${proof:+--run-proof '$proof'} > '$OUT/e41.txt' 2>&1; echo \"e41 rc=\$?\" >> '$OUT/e41.txt'" \
      > /dev/null 2>&1 < /dev/null &
  fi
  post "GPU $TAG finished rc=$rc; checker: $(tr '\n' ' ' < "$OUT/rc_checker.txt" 2>/dev/null); E41 detached ($OUT/e41.txt); out $OUT"
  exit "$rc"
fi
# ---- locked: the GPU arms ----
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || die "--locked runs only inside with_gpu_lock"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
OUT=$L/gpu_${TAG}_${GA_STAMP:?}
R0=$(rnow) || exit 7; T0=$(date +%s)
START=$(date -u -d "${GA_START_UTC:?}" +%s); END=$(date -u -d "${GA_END_UTC:?}" +%s)
(( R0 >= START )) || die "too early (window ${GA_START_UTC})" 4
(( END - R0 >= CAP_ARM_S + 120 )) || die "only $(( (END - R0) / 60 )) min left (< one arm cap $(( CAP_ARM_S / 60 )) min)" 6
[ -e "$OUT" ] && die "output root exists $OUT"
mkdir -p "$OUT"
timeout 20 journalctl -k --since "-6h" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_before.txt"
timeout 20 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$OUT/nvidia_before.csv" 2>&1
timeout 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$OUT/nvidia_before.csv" 2>&1
CACHE=$OUT/cache; mkdir -p "$CACHE/jax" "$CACHE/triton" "$CACHE/cuda" "$CACHE/autotune" "$CACHE/tmp"
rc=0
finish() { timeout 20 journalctl -k --since "@$T0" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_during.txt";
  post "GPU END ($TAG) rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); out $OUT; $(cat "$OUT"/rc_*.txt 2>/dev/null | tr '\n' ' ')"; }
trap finish EXIT
post "GPU START ($TAG) arms $ARMS from export $SRC_SHA (root ring-0 WRF walk), cap $(( CAP_ARM_S / 60 )) min/arm, window end $(date -u -d @$END +%TZ)"
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 PYTHONPATH="$EXP/src"
export GPUWRF_JAX_CACHE_DIR="$CACHE/jax" JAX_COMPILATION_CACHE_DIR="$CACHE/jax" TRITON_CACHE_DIR="$CACHE/triton"
export CUDA_CACHE_PATH="$CACHE/cuda" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$CACHE/autotune" TMPDIR="$CACHE/tmp" OMP_NUM_THREADS=4
unset XLA_FLAGS CUDA_ROOT GPUWRF_ROOT_SPEC_STAGE_CLOCK
for arm in ${ARMS//,/ }; do
  NOW=$(rnow) || exit 7
  (( END - NOW >= CAP_ARM_S + 60 )) || { echo "skip $arm: $(( END - NOW )) s left" >> "$OUT/skipped.txt"; break; }
  A=$OUT/$arm; mkdir -p "$A"; ACASE=$CASE
  case $arm in MINI|WALK) export GPUWRF_ROOT_SPEC_STAGE_CLOCK=walk;; SEED) export GPUWRF_ROOT_SPEC_STAGE_CLOCK=walk; ACASE=$SEEDCASE;;
    REL) export GPUWRF_ROOT_SPEC_STAGE_CLOCK=0;; ON1) export GPUWRF_ROOT_SPEC_STAGE_CLOCK=1;; ON2) export GPUWRF_ROOT_SPEC_STAGE_CLOCK=2;; esac
  env | grep -E '^(GPUWRF_|JAX_|XLA_|CUDA_|TRITON_|PYTHONPATH|OMP_)' | sort > "$A/env.txt"
  timeout --signal=TERM --kill-after=60 "$CAP_ARM_S" python -m gpuwrf.cli run --input-dir "$ACASE" --output-dir "$A/wrfout" \
    --domains-from-namelist $EIH --hours "$(hours_of "$arm")" > "$A/cli.log" 2>&1 &
  CPID=$!
  JAX_PLATFORMS=cpu $CPUS python "$TOOLS/stream_extract.py" "$A/wrfout" "$A/frames" --pid "$CPID" --poll 5 > "$A/extract.log" 2>&1 &
  XPID=$!
  nvidia-smi dmon -s mu -o T -d 5 > "$A/dmon.txt" 2>&1 & DPID=$!
  wait "$CPID"; r=$?; kill "$DPID" 2>/dev/null
  wait "$XPID"; x=$?
  echo "$arm rc=$r extract_rc=$x" > "$OUT/rc_$arm.txt"; (( r != 0 || x != 0 )) && rc=$(( r != 0 ? r : x ))
  (( r != 0 )) && break
done
exit "$rc"
