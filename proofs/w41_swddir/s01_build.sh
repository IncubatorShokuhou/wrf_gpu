#!/bin/bash
# S01: build + run the pristine RRTMG_SWRAD driver on the Swiss day columns (AE01/RE01 libwrflib recipe; QUIET-guarded).
#   Regenerate: python3 s01_build_inputs.py $S01_DIR && bash s01_build.sh && python3 s01_parse.py $S01_DIR <fixture dir>
#   S01_CORES (default 13,29) = the lane's granted cores.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
D=${S01_DIR:-<USER_HOME>/wrf_gpu2_lanes/w41-swddir/S01}; W=<USER_HOME>/src/wrf_pristine/WRF
CORES=${S01_CORES:-13,29}
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"; cp "$SRC/s01_driver.F90" "$D/"
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
LIBS=("$W/main/libwrflib.a" "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a"
      "$W/external/io_grib_share/libio_grib_share.a" "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time
      "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf
      -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib")
taskset -c "$CORES" nice -n 19 $FC -c $FLAGS $INC s01_driver.F90 -o s01_driver.o
taskset -c "$CORES" nice -n 19 $FC $FLAGS s01_driver.o "${LIBS[@]}" -o s01_driver.exe
mkdir -p run; cd run
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN RRTMG_SW_DATA CAMtr_volume_mixing_ratio \
         ozone.formatted ozone_lat.formatted ozone_plev.formatted \
         aerosol.formatted aerosol_lat.formatted aerosol_lon.formatted aerosol_plev.formatted; do
  for src in "$W/test/em_real/oracle_run_v090/$f" "$W/test/em_real/oracle_run/$f" "$W/run/$f"; do
    if [ -e "$src" ]; then ln -sf "$src" .; break; fi
  done
done
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do [ -e "$f" ] || { echo "MISSING $f (E160)"; exit 4; }; done
taskset -c "${CORES%%,*}" nice -n 19 ../s01_driver.exe ../s01_input.bin ../s01_output.bin
sha256sum ../s01_input.bin ../s01_output.bin ../s01_output.bin.radii
echo BUILD+RUN OK
