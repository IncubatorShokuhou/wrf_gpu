#!/bin/bash
# Build the pristine-WRF FDDA oracle in a LANE DIR copy (E104), REAL and fp64 variants.
# usage: build.sh <workdir>
set -euo pipefail
WORK=${1:?workdir}
FC=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}
PRISTINE=${PRISTINE:-<USER_HOME>/src/wrf_pristine/WRF}
HERE=$(cd "$(dirname "$0")" && pwd)
# WRF configure.wrf: FCOPTIM=-O2 -ftree-vectorize -funroll-loops, FORMAT_FREE, no -march (no FMA).
FLAGS="-O2 -ftree-vectorize -funroll-loops -ffree-form -ffree-line-length-none -w"
mkdir -p "$WORK/r4" "$WORK/r8"
cp "$PRISTINE/phys/module_fdda_psufddagd.F" "$WORK/module_fdda_psufddagd.F"
# SFDDAGD (not called, grid_sfdda=0) USEs module_model_constants: pristine copy, EM_CORE=1 cpp.
/lib/cpp -P -nostdinc -DEM_CORE=1 "$PRISTINE/share/module_model_constants.F" > "$WORK/module_model_constants.f90"
sha256sum "$WORK/module_fdda_psufddagd.F" "$PRISTINE/share/module_model_constants.F" > "$WORK/pristine_sha256.txt"
for kind in r4 r8; do
  extra=""; [ "$kind" = r8 ] && extra="-fdefault-real-8 -fdefault-double-8"
  (cd "$WORK/$kind" && cp ../module_fdda_psufddagd.F module_fdda_psufddagd.f90 && \
   $FC $FLAGS $extra -c ../module_model_constants.f90 && \
   $FC $FLAGS $extra -c module_fdda_psufddagd.f90 && \
   $FC $FLAGS $extra -c "$HERE/fdda_oracle_driver.F90" && \
   $FC -o fdda_oracle fdda_oracle_driver.o module_fdda_psufddagd.o module_model_constants.o)
done
echo built "$WORK"/r4/fdda_oracle "$WORK"/r8/fdda_oracle
