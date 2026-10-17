#!/bin/bash
# Only component routines are linked/called; this never invokes solve_em or its shared E104 dump.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
DRIVER=${1:-sms_driver}
case "$DRIVER" in sms_driver|sms_upstream_driver|sms_full_driver|sms_probe_driver|sms_stage_driver) ;; *) exit 2 ;; esac
D=<USER_HOME>/wrf_gpu2_lanes/s1-km5/oracle
W=<USER_HOME>/src/wrf_pristine/WRF
BUILD_ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build
FC=$BUILD_ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"
cp "$SRC/$DRIVER.F90" "$D/$DRIVER.F90"
# Private copies of the compiled reference and module files, E104; never run the shared wrf.exe.
cp "$W/main/libwrflib.a" "$D/libwrflib.a"
cp "$W/frame/module_configure.mod" "$W/frame/module_state_description.mod" \
   "$W/dyn_em/module_em.mod" "$W/dyn_em/module_diffusion_em.mod" "$W/share/module_bc.mod" "$W/frame/module_model_constants.mod" "$D/"
cd "$D"
FLAGS="-w -ffree-line-length-none -O2"
taskset -c 13 nice -n 19 "$FC" -c $FLAGS -I"$D" "$DRIVER.F90" -o "$DRIVER.o"
taskset -c 13 nice -n 19 "$FC" $FLAGS "$DRIVER.o" "$D/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$BUILD_ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$BUILD_ENV/lib" -o "$DRIVER.exe"
sha256sum "$DRIVER.F90" libwrflib.a > "$DRIVER-reference-sha256.txt"
echo "BUILD OK $D/$DRIVER.exe"
