#!/usr/bin/env bash
# Build the pristine-linked mym_condensation oracle (links the UNEDITED pristine build objects).
# Copied from lane o5-n333 proofs/o5_n333/oracle/build.sh (CPDF01 form, mym_condensation only).
set -euo pipefail
W=${WRF_PRISTINE:-<USER_HOME>/src/wrf_pristine/WRF}
FC=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran}
OUT=${1:?out dir}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
"$FC" -I"$W/phys" -ffree-form -ffree-line-length-none -O2 -c -o "$OUT/drv.o" "$HERE/mym_condensation_driver.F90" -J "$OUT"
"$FC" -o "$OUT/mym_condensation_oracle" "$OUT/drv.o" "$W/phys/module_bl_mynnedmf.o" \
   "$W/phys/module_bl_mynnedmf_common.o" "$W/phys/ccpp_kind_types.o" "$W/share/module_model_constants.o"
sha256sum "$W/phys/module_bl_mynnedmf.o" "$OUT/mym_condensation_oracle"
