#!/usr/bin/env bash
# RRTMG Tier-1 bisection driver (lane o5-port, manager 23:19Z). Sequential arms, one process each (E105/E119).
# usage: rrtmg_tier1_bisect.sh OUT.jsonl PHASE   (PHASE = loo | addin FLAG=VAL...)
set -u
OUT=$1; PHASE=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 TMPDIR=${TMPDIR:-/home/wz/o5port/t1tmp}
mkdir -p "$TMPDIR"
arm() { local label=$1; shift; env "$@" python "$HERE/rrtmg_tier1_arm.py" "$label" "$OUT" || echo "ARM_FAIL $label"; }
if [ "$PHASE" = loo ]; then
  arm release
  arm legacy GPUWRF_FAST_DEFAULTS=0
  for kv in GPUWRF_MCICA_JUMPAHEAD=0 GPUWRF_MCICA_LEGACY_FP64=1 GPUWRF_RRTMG_LW_FUSED_TRANSFER=0 \
            GPUWRF_RRTMG_SW_FUSED_QUADRATURE=0 GPUWRF_RRTMG_COLUMN_TILE_COLS=1024 GPUWRF_RRTMG_LW_BAND_SUMS=0 \
            GPUWRF_MCICA_KISS_KERNEL=0 GPUWRF_RRTMG_SW_BAND_SUMS=0 GPUWRF_RRTMG_MAXRAND=0 GPUWRF_RRTMG_MP_RE=0 \
            GPUWRF_SOLAR_JULIAN_WRF=0 GPUWRF_RRTMG_REAL_CONSTANTS=0 GPUWRF_RRTMG_SW_REFTRA_WRF=0 \
            GPUWRF_THOMPSON_WRF_CONSTANTS=0 GPUWRF_HISTORY_SWDOWN_HORIZONTAL=0; do
    arm "loo:$kv" "$kv"
  done
  arm "loo:REAL_ALL_PAIR=0" GPUWRF_DYN_REAL_ALL=0 GPUWRF_CARRY_REAL_ALL=0
else
  for kv in "$@"; do arm "addin:$kv" GPUWRF_FAST_DEFAULTS=0 "$kv"; done
fi
echo BISECT_DONE
