#!/usr/bin/env bash
# E104: build in a lane-owned directory, never in the pristine source tree.
set -euo pipefail
work=${1:?lane build directory}
pristine=${PRISTINE:-<USER_HOME>/src/wrf_pristine/WRF}
fc=${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work/r4" "$work/r8"
cp "$pristine/phys/module_fdda_spnudging.F" "$work/"
cat > "$work/stubs.f90" <<'EOF'
module module_wrf_error
  character(len=512) :: wrf_err_message
end module
module module_state_description
end module
module module_domain
  type domain
    real, allocatable :: dif_analysis(:,:,:)
  end type
end module
subroutine wrf_message(s)
  character(len=*) :: s
end subroutine
subroutine da_wrf_message(s)
  character(len=*) :: s
end subroutine
subroutine da_wrf_error_fatal(s)
  character(len=*) :: s
  write(0,*) 'FFT FATAL: ',s
  stop 3
end subroutine
subroutine wrf_error_fatal(s)
  character(len=*) :: s
  write(0,*) 'FATAL: ',s
  stop 3
end subroutine
subroutine get_wrf_debug_level(level)
  integer :: level
  level=0
end subroutine
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor=.false.
end function
EOF
# Compile only the FFTPACK5 dependency closure, from unchanged source copies.
fft=(rfftmi rfftmf rfftmb mrfti1 mrftf1 mrftb1 mradf2 mradf3 mradf4 mradf5 mradfg mradb2 mradb3 mradb4 mradb5 mradbg xerfft xercon)
for name in "${fft[@]}"; do
  cp "$pristine/external/fftpack/fftpack5/$name.F" "$work/$name.f90"
done
sha256sum "$work/module_fdda_spnudging.F" "$work/"*.f90 > "$work/sources.sha256"
for kind in r4 r8; do
  flags=(-O2 -ftree-vectorize -funroll-loops -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
  # FFTPACK declares explicit REAL(kind=4), unlike WRF's default REAL.
  # Promote those too for the optional REAL8 diagnostic (never the REAL4 truth).
  if [ "$kind" = r8 ]; then flags+=(-fdefault-real-8 -fdefault-double-8 -freal-4-real-8); fi
  (
    cd "$work/$kind"
    "$fc" "${flags[@]}" -c ../stubs.f90
    /lib/cpp -P -nostdinc ../module_fdda_spnudging.F > spectral.f90
    "$fc" "${flags[@]}" -c spectral.f90
    for name in "${fft[@]}"; do "$fc" "${flags[@]}" -c "../$name.f90"; done
    "$fc" "${flags[@]}" -c "$here/spectral_oracle.F90"
    "$fc" -o spectral_oracle ./*.o
  )
done
