#!/usr/bin/env bash
# t-cfix-pblmix: UNMODIFIED WRF phys/module_bl_myjpbl.F (MYJPBL) on REAL cloudy CPU-WRF columns
# (c03 = mp2 Lin + MYJ: real QICE and TKE_PBL). Builds DOUBLE (-fdefault-real-8) and REAL (WRF REAL4);
# QCI present (T: flQI, the WRF caller always passes QI_CURR) and absent (F: NSPEC=3).
# Usage: build_myj_moist_oracle.sh <build_dir> <wrfout[,...]> <ncol per file>
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
B="${1:?build dir}"; WRFOUT="${2:?wrfout}"; NCOL="${3:-8}"
rm -rf "$B"; mkdir -p "$B"
W=<USER_HOME>/src/wrf_pristine/WRF
cp $W/share/module_model_constants.F $W/phys/module_bl_myjpbl.F "$HERE/myj_moist_driver.f90" "$B/"
cd "$B"; set -e
( cd "$HERE" && python3 myj_columns.py "$WRFOUT" "$NCOL" "$B/cols.txt" "$B/inputs.json" )
sha256sum module_model_constants.F module_bl_myjpbl.F myj_moist_driver.f90 cols.txt > sources.sha256
F="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fno-fast-math"
for mode in r8 r4; do
  X=""; [ "$mode" = r8 ] && X="-fdefault-real-8 -fdefault-double-8"
  mkdir -p "$mode"; ( cd "$mode"
    $FC $F $X -c ../module_model_constants.F; $FC $F $X -c ../module_bl_myjpbl.F
    $FC $F $X -c ../myj_moist_driver.f90
    $FC $F $X -o myj_oracle module_model_constants.o module_bl_myjpbl.o myj_moist_driver.o
    for fq in T F; do ./myj_oracle ../cols.txt "$fq" > "out_${fq}.txt"; done )
done
python3 "$HERE/make_fixture.py" "$B" "$HERE/myj_moist_fixture.json" r8_qi=r8/out_T.txt r8_noqi=r8/out_F.txt r4_qi=r4/out_T.txt
echo OK
