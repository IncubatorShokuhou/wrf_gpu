#!/usr/bin/env bash
# GPU smoke of `gpuwrf optimize` (lane w41-optimize, <= 20 GPU-min; manager-granted window + lease only).
# usage: gpu_smoke.sh ROOT [BUNDLE_CONSUMER|none]   (default consumer: scripts/release/bundle_consumer.sh of this tree)
#   Arm A (<= 9 min, pauses instead of overrunning; resume later with --resume): optimize run on the GPU, Swiss d01, released c00 + c36, 2 rungs + held-out, 3 h members,
#          SYNTHETIC relabelled dates, --max-parallel 4; with a consumer: --bundle v0.4.0 (c00 Swiss is primed in the
#          bundle, c36 compiles into the private layer). Proves cross-config batches, rung receipts, bundle never written.
#   Arm B (<= 12 min): d01-only VRAM/host probe on the WN3 grid (c60 input: WN3 0227 d01, sst_update 1): released
#          cu0 / cu1 (KF) members in PARALLEL with a deliberately small 1.0 GiB pool, so the per-process
#          nvidia-smi peak shows the real need (cuda_async grows past a small pool) and VmHWM the host need.
# Afterwards (CPU): verify A, summarise B, E41 cubin gate of B's cache, delete caches.
set -uo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
ROOT=$1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONSUMER=${2:-$REPO/scripts/release/bundle_consumer.sh}
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
C60=<USER_HOME>/wrf_gpu2_lanes/v-prep/c60/input
LOCK="$REPO/scripts/with_gpu_lock.sh --label w41-optimize"
# Manager rule 2026-10-17 11:37Z: never nest with_ram_slot and the GPU lock; cap the GPU process itself.
CAP="systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 --"
cpu() { JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$REPO/src" taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python3 "$@"; }
mkdir -p "$ROOT" || exit 2
test ! -e "$ROOT/A" || { echo "exists: $ROOT/A"; exit 2; }
# --- CPU preparation (before the lock) ---
cpu "$REPO/proofs/w41_optimize/cpu_dryrun.py" prepare "$ROOT/A" 3 || exit 2
for cu in 0 1; do
  d=$ROOT/B/cases/cu$cu; mkdir -p "$d"
  for f in wrfinput_d01 wrfbdy_d01 wrflowinp_d01; do ln -s "$C60/$f" "$d/$f"; done
  sed -E "s/^( cu_physics *= *)[0-9]+,/\1$cu,/" "$C60/namelist.input" > "$d/namelist.input"
  grep -q "cu_physics *= *$cu," "$d/namelist.input" || { echo "cu edit failed"; exit 2; }
done
echo "== CPU prep done $(date -u +%FT%TZ)"
# --- Arm A: optimize on the GPU ---
bundle=()
if [ "$CONSUMER" != none ]; then bundle=(--bundle "$P" --bundle-consumer "$CONSUMER"); else
  export GPUWRF_JAX_CACHE_DIR="$ROOT/A/cache/jax" JAX_COMPILATION_CACHE_DIR="$ROOT/A/cache/jax" \
         GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$ROOT/A/cache/autotune"; fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
$LOCK -- timeout 720 $CAP env -u JAX_PLATFORMS PYTHONPATH="$REPO/src" \
  python3 -m gpuwrf.cli optimize run --study "$ROOT/A/study.json" --output "$ROOT/A/run" --max-parallel 4 \
  --host-gb-per-member 8 --chunk-minutes 9 "${bundle[@]}"
echo "== arm A rc=$? $(date -u +%FT%TZ)"
unset GPUWRF_JAX_CACHE_DIR JAX_COMPILATION_CACHE_DIR GPUWRF_XLA_AUTOTUNE_CACHE_DIR
# --- Arm B: WN3 d01-only members in parallel, small pool ---
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
$LOCK -- timeout 720 $CAP env -u JAX_PLATFORMS PYTHONPATH="$REPO/src" GPUWRF_JAX_CACHE_DIR="$ROOT/B/cache/jax" \
  JAX_COMPILATION_CACHE_DIR="$ROOT/B/cache/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$ROOT/B/cache/autotune" \
  "$REPO/scripts/run_parallel_cases.sh" --out-root "$ROOT/B/out" --pool-gib 1.0 --max-parallel 2 --host-gb-per-case 8 \
  --vram-sample-interval 1 "$ROOT"/B/cases/cu0 "$ROOT"/B/cases/cu1 -- --domains-from-namelist --hours 3
echo "== arm B rc=$? $(date -u +%FT%TZ)"
# --- CPU afterwards ---
cpu "$REPO/proofs/w41_optimize/cpu_dryrun.py" verify "$ROOT/A" "$ROOT/A/GPU_ARM_A.json"
cpu - "$ROOT" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
rows = {}
for rec in sorted((root/"B"/"out").glob("*/receipt.json")):
    r = json.loads(rec.read_text())
    rows[r["name"]] = {k: r.get(k) for k in ("rc", "wall_s", "vram_peak_mib", "vmhwm_kb", "concurrent_at_start",
                                             "aot_cache_warm", "allocator", "note")}
run = json.loads((root/"B"/"out"/"parallel_run.json").read_text())
(root/"B"/"GPU_ARM_B.json").write_text(json.dumps({"members": rows, "vram_total_bytes": run.get("vram_total_bytes"),
    "vram_free_bytes_at_start": run.get("vram_free_bytes_at_start"), "events": run.get("events")}, indent=1)+"\n")
print(json.dumps(rows))
EOF
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
cpu "$REPO/scripts/wave_cubin_gate.py" "$ROOT/B/cache/jax" "$ROOT/B/e41" || echo "E41 gate rc=$?"
if [ "$CONSUMER" = none ]; then cpu "$REPO/scripts/wave_cubin_gate.py" "$ROOT/A/cache/jax" "$ROOT/A/e41" || echo "E41 A rc=$?"
elif [ -d "$ROOT/A/run/bundle_layer/upper/jax" ]; then
  cpu "$REPO/scripts/wave_cubin_gate.py" "$ROOT/A/run/bundle_layer/upper/jax" "$ROOT/A/e41" || echo "E41 A rc=$?"; fi
echo "== done $(date -u +%FT%TZ); caches to delete after review: $ROOT/A/cache $ROOT/A/run/bundle_layer $ROOT/B/cache"
