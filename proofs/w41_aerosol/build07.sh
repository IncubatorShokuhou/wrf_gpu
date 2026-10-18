#!/bin/bash
# AE07: build + run the pristine WRF aer_opt=3 oracle (same libwrflib recipe and run dir as build.sh / AE01).
#   Regenerate: python3 ae01_build_inputs.py $AE01_DIR && python3 ae07_build_inputs.py $AE01_DIR && bash build07.sh
#               && python3 ae07_parse.py $AE01_DIR <fixture dir>
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
D=${AE01_DIR:-<USER_HOME>/wrf_gpu2_lanes/w41-aerosol/AE01}; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"; cp "$SRC/ae07_driver.F90" "$D/"
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
LIBS=("$W/main/libwrflib.a" "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a"
      "$W/external/io_grib_share/libio_grib_share.a" "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time
      "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf
      -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib")
taskset -c 8,9,12,13 nice -n 19 $FC -c $FLAGS $INC ae07_driver.F90 -o ae07_driver.o
taskset -c 8,9,12,13 nice -n 19 $FC $FLAGS ae07_driver.o "${LIBS[@]}" -o ae07_driver.exe
[ -d run ] || { echo "run dir missing: run build.sh first"; exit 2; }
cd run
taskset -c 8 nice -n 19 ../ae07_driver.exe ../ae07_input.bin ../ae07_output.bin
sha256sum ../ae07_input.bin ../ae07_output.bin
echo BUILD+RUN OK
