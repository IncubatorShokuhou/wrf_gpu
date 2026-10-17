#!/usr/bin/env bash
# Caller supplies QUIET guard, affinity, nice, CPU-only env and timeout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$1"; shift
WRF_PHYS="${WRF_PRISTINE_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}/phys"
export PATH="<USER_HOME>/miniconda3/envs/wrfbuild/bin:${PATH}"
ulimit -c 0
mkdir -p "$WORK"
for prec in r4 r8; do
  B="$WORK/build_$prec"; mkdir -p "$B"; cd "$B"
  for m in module_gfs_machine module_gfs_physcons module_gfs_funcphys module_cu_sas; do
    cp "$WRF_PHYS/$m.F" .
  done
  cp "$HERE/sas94_oracle.F90" .
  BASE="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0"
  if [ "$prec" = r8 ]; then FL="$BASE -fdefault-real-8 -fdefault-double-8"; else FL="$BASE"; fi
  printf '%s\n' "$FL" > flags.txt
  for m in module_gfs_machine module_gfs_physcons module_gfs_funcphys module_cu_sas; do
    gfortran $FL -c "$m.F"
  done
  gfortran $FL -c sas94_oracle.F90
  gfortran -O2 -o sas94_oracle module_gfs_machine.o module_gfs_physcons.o module_gfs_funcphys.o module_cu_sas.o sas94_oracle.o
  sha256sum "$WRF_PHYS/module_cu_sas.F" module_cu_sas.F > sources.sha256
  for inp in "$@"; do
    stem="$(basename "$inp" .txt)"
    ./sas94_oracle < "$inp" > "$WORK/${stem}_$prec.out"
  done
done
