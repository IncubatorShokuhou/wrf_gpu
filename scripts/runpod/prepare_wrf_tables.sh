#!/usr/bin/env bash
# Called by setup.sh; separately testable without CUDA, pip installs or a CPU-WRF build.
set -euo pipefail
wrf=${1:?WRF checkout required}
expected=f52c197ed39d12e087d02c50f412d90d418f6186
actual=$(git -C "$wrf" rev-parse HEAD)
if [ "$actual" != "$expected" ]; then echo "WRF table source must be pinned v4.7.1 ($expected); got $actual" >&2; exit 1; fi
git -C "$wrf" submodule update --init --depth 1 phys/noahmp
# WRF's build would normally create this link. Runtime-only installations need it explicitly.
ln -sfn ../phys/noahmp/parameters/MPTABLE.TBL "$wrf/run/MPTABLE.TBL"
for name in MPTABLE.TBL SOILPARM.TBL GENPARM.TBL VEGPARM.TBL RRTMG_LW_DATA RRTMG_SW_DATA; do
  if [ ! -s "$wrf/run/$name" ]; then echo "missing/empty WRF runtime table: $wrf/run/$name" >&2; exit 1; fi
done
echo "WRF v4.7.1 runtime tables OK (including pinned Noah-MP submodule)"
