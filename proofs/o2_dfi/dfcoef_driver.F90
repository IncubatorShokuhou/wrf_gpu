! Driver for the VERBATIM pristine WRF 4.7.1 share/dfi.F filter routines (dfcoef .. cnr).
! Reproduces optfil_driver (share/dfi.F:775-831) for a list of (nfilter, rundfi, dt, tauc)
! configurations and prints every hcoeff(1:2*nstep2) as the exact IEEE binary32 bit pattern.
! Input (stdin): lines "nfilter rundfi dt tauc"  (rundfi integer seconds, dt/tauc REAL).
subroutine wrf_message(str)
  character(len=*), intent(in) :: str
end subroutine wrf_message

program dfcoef_driver
  implicit none
  integer, parameter :: ndim = 1000          ! Registry dfi_time_dim default
  real :: hcoeff(ndim), dt, timestep, tauc
  integer :: nfilter, rundfi, nstep2, i, ios, nout
  do
    read(*, *, iostat=ios) nfilter, rundfi, dt, tauc
    if (ios /= 0) exit
    ! --- optfil_driver body (dfi.F:798-829), grid%dt -> dt, grid%dfi_cutoff_seconds -> tauc ---
    timestep = abs(dt)
    rundfi = abs(rundfi)
    nstep2 = ceiling((1.0 + real(rundfi)/timestep) / 2.0)
    hcoeff(:) = 0.0
    call dfcoef(nstep2-1, dt, tauc, nfilter, hcoeff)
    if ( mod(int(1.0 + real(rundfi)/timestep), 2) /= 0 ) then
      do i = 1, nstep2-1
        hcoeff(2*nstep2-i) = hcoeff(i)
      end do
    else
      do i = 1, nstep2
        hcoeff(2*nstep2-i+1) = hcoeff(i)
      end do
    end if
    nout = 2*nstep2
    write(*, '(A,I4,1X,I8,1X,Z8.8,1X,Z8.8,1X,I6)') 'CFG ', nfilter, rundfi, transfer(dt, 1), transfer(tauc, 1), nout
    do i = 1, nout
      write(*, '(Z8.8)') transfer(hcoeff(i), 1)
    end do
  end do
end program dfcoef_driver
