#!/bin/bash
# AE01: build + run the pristine WRF aer_opt=1 oracles (RE01/TH08 libwrflib recipe; lane cores; QUIET-guarded).
#   AE01_DIR (default the w41-aerosol lane dir) receives objects, run dir (read-only table links) and outputs.
#   Regenerate: python3 ae01_build_inputs.py $AE01_DIR && bash build.sh && python3 ae01_parse.py $AE01_DIR <fixture dir>
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
D=${AE01_DIR:-<USER_HOME>/wrf_gpu2_lanes/w41-aerosol/AE01}; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"; cp "$SRC/ae01_driver.F90" "$SRC/ae01_units.f90" "$D/"
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
LIBS=("$W/main/libwrflib.a" "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a"
      "$W/external/io_grib_share/libio_grib_share.a" "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time
      "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf
      -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib")
for prog in ae01_driver ae01_units; do
  src=$prog.F90; [ -e "$src" ] || src=$prog.f90
  taskset -c 8,9,12,13 nice -n 19 $FC -c $FLAGS $INC "$src" -o $prog.o
  taskset -c 8,9,12,13 nice -n 19 $FC $FLAGS $prog.o "${LIBS[@]}" -o $prog.exe
done
mkdir -p run; cd run
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN RRTMG_SW_DATA CAMtr_volume_mixing_ratio \
         ozone.formatted ozone_lat.formatted ozone_plev.formatted \
         aerosol.formatted aerosol_lat.formatted aerosol_lon.formatted aerosol_plev.formatted; do
  for src in "$W/test/em_real/oracle_run_v090/$f" "$W/test/em_real/oracle_run/$f" "$W/run/$f"; do
    if [ -e "$src" ]; then ln -sf "$src" .; break; fi
  done
done
taskset -c 8 nice -n 19 ../ae01_units.exe ../ae01_units_input.bin ../ae01_units_output.bin
taskset -c 8 nice -n 19 ../ae01_driver.exe ../ae01_input.bin ../ae01_output.bin
sha256sum ../ae01_*input.bin ../ae01_*output.bin
echo BUILD+RUN OK
