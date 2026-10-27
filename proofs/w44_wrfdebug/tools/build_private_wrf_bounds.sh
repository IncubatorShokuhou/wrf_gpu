#!/bin/bash
# Private relink of the gen2 dmpar wrf.exe with ONE recompiled object (dyn_em/start_em.F).
# Usage: build_private_wrf.sh <outdir> <start_em.F source>   (never writes into the original build)
set -euo pipefail
OUT=$1; SRC=$2
B=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/_build_gen2_dmpar
FC=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/gfortran
F=$B/CMakeFiles/WRF_Core.dir/flags.make
mkdir -p "$OUT/mod"
get() { sed -n "s/^$1 = //p" "$F"; }
DEFS=$(get Fortran_DEFINES); INCS=$(get Fortran_INCLUDES)
FLAGS=$(get Fortran_FLAGS | sed "s#-J[^ ]*#-J$OUT/mod#")
cd "$OUT"
$FC $DEFS $INCS $FLAGS -ffree-form -cpp -O2 -ftree-vectorize -funroll-loops -ffree-line-length-none -fallow-argument-mismatch -fallow-invalid-boz -ffree-line-length-none -fcheck=bounds -g -c "$SRC" -o "$OUT/start_em.F.o"
cp "$B/libWRF_Core.a" "$OUT/libWRF_Core.a"
cp "$OUT/start_em.F.o" "$OUT/member/start_em.F.o" 2>/dev/null || { mkdir -p "$OUT/member"; cp "$OUT/start_em.F.o" "$OUT/member/start_em.F.o"; }
( cd "$OUT/member" && ar rD "$OUT/libWRF_Core.a" start_em.F.o )
LINK=$(cat $B/main/CMakeFiles/wrf.dir/link.txt)
LINK=${LINK//CMakeFiles\/wrf.dir\/link.d/$OUT\/link.d}
LINK=${LINK//CMakeFiles\/wrf.dir\/wrf.F.o/$B\/main\/CMakeFiles\/wrf.dir\/wrf.F.o}
LINK=${LINK//CMakeFiles\/wrf.dir\/module_wrf_top.F.o/$B\/main\/CMakeFiles\/wrf.dir\/module_wrf_top.F.o}
LINK=${LINK//..\/libWRF_Core.a/$OUT\/libWRF_Core.a}
LINK=${LINK//..\/external/$B\/external}
LINK=${LINK//-o wrf /-o $OUT\/wrf.exe }
echo "$LINK" > "$OUT/link_cmd.txt"
eval "$LINK"
sha256sum "$OUT/start_em.F.o" "$OUT/wrf.exe" "$SRC"
