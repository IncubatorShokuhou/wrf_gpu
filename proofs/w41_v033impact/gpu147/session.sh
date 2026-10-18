#!/usr/bin/env bash
# w41-v033impact #147 — IMMUTABLE GPU session (one lock). Never edit after launch (E163): re-launch a new copy instead.
# Question: does a v0.3.3 fused d02/d03 program compiled for date A (0227) change date B (0707) outputs when it is
# (wrongly) loaded from a shared cache under the date-blind v4 cheap key?
# Arms (v0.3.3 tag 86e6303702, CLI, --domains-from-namelist, all arms on ONE cache path string $LIVE so every
# GPUWRF_* path in the cheap key is identical (E32/E143); each arm's cache dir is renamed in/out of $LIVE):
#   prime: 0227 3 h, empty cache A          -> cold d01 + fused(0227 constants) blobs
#   P    : 0707 24 h, cache A               -> expect d01 hit + fused cheap-key hit on the 0227 blob (wrong load)
#   F    : 0707 24 h, cache B = copy of A after P minus aot/*/fused_d02 -> same d01 blob, fused compiled fresh (0707)
#   V    : 0707 24 h, cache C = copy of A after P, GPUWRF_AOT_VERIFY=1 -> verify mismatch -> fresh compile
#          (= v0.3.3 mitigation test + compile-noise control for F)
set -u
export LC_ALL=C
L=<USER_HOME>/wrf_gpu2_lanes/w41-v033impact
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-v033impact
SNAP=$L/snap_v033
SNAP_SHA=86e63037020392df38c3f600a9d903e258e3e9d4
G=$S/gpu147
LIVE=$S/cache_live
DEADLINE=$(date -u -d "2026-10-18T12:50:00Z" +%s)
mkdir -p "$G"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$G/session.log"; }

[[ "${GPUWRF_GPU_LOCK_HELD:-}" == 1 ]] || { log "REFUSE: GPU lock not held"; exit 3; }
[[ "$(git -C "$SNAP" rev-parse HEAD)" == "$SNAP_SHA" ]] || { log "REFUSE: snapshot HEAD"; exit 3; }
[[ -z "$(git -C "$SNAP" status --porcelain)" ]] || { log "REFUSE: snapshot dirty"; exit 3; }
[[ ! -e "$LIVE" ]] || { log "REFUSE: $LIVE exists"; exit 3; }
for d in wn3_0227 g2_20260707; do [[ -s "$S/cases/$d/wrfinput_d03" ]] || { log "REFUSE: case $d"; exit 3; }; done
log "session start label=${GPUWRF_GPU_LOCK_LABEL:-?} snap=$SNAP_SHA df_mnt_data=$(df -BG --output=avail <DATA_ROOT> | tail -1)"
cd "$SNAP" || exit 3
env PYTHONPATH="$SNAP/src" python -c "import gpuwrf, jax; d=jax.devices()[0]; print('PREFLIGHT', gpuwrf.__file__, gpuwrf.__version__, d.platform, d.device_kind); assert d.platform=='gpu' and gpuwrf.__file__.startswith('$SNAP/src')" >> "$G/session.log" 2>&1 \
  || { log "REFUSE: preflight"; exit 3; }

inventory() {  # sha256 of every AOT blob/meta in the live cache
  find "$LIVE/aot" -type f \( -name '*.xlaexec' -o -name '*.meta' -o -name '*.quarantine' \) -printf '%P %s\n' 2>/dev/null | sort > "$1.list"
  (cd "$LIVE/aot" 2>/dev/null && find . -type f -name '*.xlaexec' -print0 | sort -z | xargs -0 -r sha256sum) > "$1.sha256"
}

arm() {  # name case hours budget_s [ENV=VAL ...]
  local name=$1 case=$2 hours=$3 budget=$4; shift 4
  local now; now=$(date +%s)
  if (( now + budget > DEADLINE )); then log "SKIP $name: budget ${budget}s would pass 12:50Z"; return 9; fi
  mkdir -p "$G/$name"
  log "ARM $name start case=$case hours=$hours budget=${budget}s env=[$*]"
  env PYTHONPATH="$SNAP/src" GPUWRF_JAX_CACHE_DIR="$LIVE" JAX_COMPILATION_CACHE_DIR="$LIVE" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$LIVE/autotune" "$@" \
      timeout -k 60 "$budget" python -m gpuwrf.cli run --input-dir "$S/cases/$case" --output-dir "$G/$name/out" \
      --scratch-dir "$G/$name/scratch" --domains-from-namelist --hours "$hours" > "$G/$name/run.log" 2>&1
  local rc=$?
  grep -a -E "nested-aot|cheap|AOT|aot" "$G/$name/run.log" | cut -c1-400 > "$G/$name/aot_lines.txt"
  inventory "$G/$name/aot_inventory"
  rm -rf -- "$G/$name/scratch"
  log "ARM $name end rc=$rc files=$(ls "$G/$name/out" 2>/dev/null | grep -c '^wrfout') du=$(du -sh "$G/$name/out" 2>/dev/null | cut -f1)"
  return $rc
}

swap_in()  { mv -T "$S/$1" "$LIVE" && log "cache $1 -> live"; }
swap_out() { mv -T "$LIVE" "$S/$1" && log "cache live -> $1 ($(du -sh "$S/$1" | cut -f1))"; }

mkdir -p "$LIVE"
arm prime wn3_0227 3 2400 || log "prime rc nonzero"
swap_out cache_A
swap_in cache_A
arm P g2_20260707 24 1500 || log "P rc nonzero"
swap_out cache_A
cp -a "$S/cache_A" "$S/cache_B" && cp -a "$S/cache_A" "$S/cache_C" && log "copied A -> B, C"
for d in "$S"/cache_B/aot/*/fused_d02; do
  [[ "$d" == <DATA_ROOT>/wrf_gpu2/scratch/w41-v033impact/cache_B/aot/*/fused_d02 && -d "$d" ]] || continue
  log "B: remove fused entries $(ls "$d" | tr '\n' ' ')"; rm -rf -- "$d"
done
swap_in cache_B
arm F g2_20260707 24 2100 || log "F rc nonzero"
swap_out cache_B
swap_in cache_C
arm V g2_20260707 24 2100 GPUWRF_AOT_VERIFY=1 || log "V rc nonzero"
swap_out cache_C
log "session end"
