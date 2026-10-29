#!/bin/bash
# Same TH08 pristine-library driver; lane cores and live read-only tables.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G02/driver
wrf=<USER_HOME>/src/wrf_pristine/WRF
compiler_env=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build
tables=$wrf/test/em_real/oracle_run_v090_mp28
mkdir -p "$out"
for name in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
  test -f "$tables/$name"
  ln -sf "$tables/$name" "$out/$name"
done
ln -sf "$wrf/run/CCN_ACTIVATE.BIN" "$out/CCN_ACTIVATE.BIN"
cp "$repo/proofs/thompson/ice_process_oracle/thompson_column_driver.F90" "$out/"
JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python - "$out/thompson_column_driver.F90" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
# WRF thompson_init :476 unconditionally formats HGT(its+1,k,jts+1)
# for a debug message, even at debug_level=0. The mp8 physics still uses
# the original single row; only init's diagnostic height halo is widened.
old='hgt3d(ncol,nlev,1))';assert s.count(old)==1
s=s.replace(old,'hgt3d(ncol,nlev,2))')
s=s.replace('  CALL thompson_init(', '  hgt3d(:,:,2) = hgt3d(:,:,1)\n  CALL thompson_init(')
start=s.index('  CALL thompson_init(');end=s.index('  CALL mp_gt_driver(',start)
init=s[start:end];assert init.count('JME=jme')==1
s=s[:start]+init.replace('JME=jme','JME=2')+s[end:]
p.write_text(s)
PY
cd "$out"
fc=$compiler_env/bin/x86_64-conda-linux-gnu-gfortran
flags=(-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2)
incs=(-I"$wrf/phys" -I"$wrf/frame" -I"$wrf/share" -I"$wrf/main" -I"$wrf/inc")
"$fc" "${flags[@]}" "${incs[@]}" -c thompson_column_driver.F90 -o thompson_column_driver.o
"$fc" "${flags[@]}" thompson_column_driver.o "$wrf/main/libwrflib.a" \
 "$wrf/external/fftpack/fftpack5/libfftpack.a" "$wrf/external/io_grib1/libio_grib1.a" \
 "$wrf/external/io_grib_share/libio_grib_share.a" "$wrf/external/io_int/libwrfio_int.a" \
 -L"$wrf/external/esmf_time_f90" -lesmf_time "$wrf/frame/module_internal_header_util.o" \
 "$wrf/frame/pack_utils.o" -L"$wrf/external/io_netcdf" -lwrfio_nf \
 -L"$compiler_env/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$compiler_env/lib" -o thompson_column_driver.exe
