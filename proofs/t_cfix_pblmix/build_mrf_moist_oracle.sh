#!/usr/bin/env bash
# t-cfix-pblmix: UNMODIFIED WRF phys/module_bl_mrf.F (MRF) on REAL cloudy c09 CPU-WRF columns.
# Two builds: DOUBLE (-fdefault-real-8, transcription gate) and REAL (WRF REAL4, operational gate);
# flag_qi=T (mp with P_QI: QI3D mixed, RQIBLTEN written) and F (no ice slot).
# Usage: build_mrf_moist_oracle.sh <build_dir> <wrfout> <ncol>
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
B="${1:?build dir}"; WRFOUT="${2:?wrfout}"; NCOL="${3:-8}"
rm -rf "$B"; mkdir -p "$B"
cp <USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_mrf.F "$HERE/mrf_moist_driver.f90" "$B/"
cd "$B"; set -e
python3 "$HERE/real_columns.py" "$WRFOUT" "$NCOL" cols.txt inputs.json
sha256sum module_bl_mrf.F mrf_moist_driver.f90 cols.txt > sources.sha256
F="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fno-fast-math"
for mode in r8 r4; do
  X=""; [ "$mode" = r8 ] && X="-fdefault-real-8 -fdefault-double-8"
  mkdir -p "$mode"; ( cd "$mode"
    $FC $F $X -c ../module_bl_mrf.F; $FC $F $X -c ../mrf_moist_driver.f90
    $FC $F $X -o mrf_oracle module_bl_mrf.o mrf_moist_driver.o
    for fq in T F; do ./mrf_oracle ../cols.txt "$fq" > "out_${fq}.txt"; done )
done

python3 "$HERE/make_fixture.py" "$B" "$HERE/mrf_moist_fixture.json" r8_qi=r8/out_T.txt r8_noqi=r8/out_F.txt r4_qi=r4/out_T.txt
echo OK
