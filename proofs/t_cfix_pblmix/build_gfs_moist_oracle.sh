#!/usr/bin/env bash
# t-cfix-pblmix: UNMODIFIED WRF phys/module_bl_gfs.F (BL_GFS -> MONINP -> TRIDI2/TRIDIN/TRIDIT) on REAL cloudy CPU-WRF columns.
# Two builds: DOUBLE (-fdefault-real-8, transcription gate) and REAL (WRF REAL4, operational gate);
# P_QI present (T: NTRAC=3, QI3D mixed) and absent (F: NTRAC=2). GFS internals are kind_phys=real*8 in both.
# Usage: build_gfs_moist_oracle.sh <build_dir> <wrfout> <ncol>
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC=<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran
B="${1:?build dir}"; WRFOUT="${2:?wrfout}"; NCOL="${3:-8}"
rm -rf "$B"; mkdir -p "$B"
W=<USER_HOME>/src/wrf_pristine/WRF/phys
cp $W/module_gfs_machine.F $W/module_gfs_physcons.F $W/module_bl_gfs.F "$HERE/gfs_moist_driver.f90" "$B/"
cd "$B"; set -e
python3 "$HERE/real_columns.py" "$WRFOUT" "$NCOL" cols.txt inputs.json
sha256sum module_gfs_machine.F module_gfs_physcons.F module_bl_gfs.F gfs_moist_driver.f90 cols.txt > sources.sha256
F="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fno-fast-math"
for mode in r8 r4; do
  X=""; [ "$mode" = r8 ] && X="-fdefault-real-8 -fdefault-double-8"
  mkdir -p "$mode"; ( cd "$mode"
    for m in module_gfs_machine module_gfs_physcons module_bl_gfs; do $FC $F $X -c ../$m.F; done
    $FC $F $X -c ../gfs_moist_driver.f90
    $FC $F $X -o gfs_oracle module_gfs_machine.o module_gfs_physcons.o module_bl_gfs.o gfs_moist_driver.o
    for fq in T F; do ./gfs_oracle ../cols.txt "$fq" > "out_${fq}.txt"; done )
done

python3 "$HERE/make_fixture.py" "$B" "$HERE/gfs_moist_fixture.json" r8_qi=r8/out_T.txt r8_noqi=r8/out_F.txt r4_qi=r4/out_T.txt
echo OK
