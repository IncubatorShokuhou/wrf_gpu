! Caller for verbatim WRF 4.7.1 cal_cldfra3 (module_radiation_driver.F:1470-1500 wiring:
! modify_qvapor=.false., max_relh=1.5, kts=1, kte=nz, debug_flag=.false.).
! stdin: stream binary  int32 ncol, int32 nz, real gridkm, then per column
!        qv(nz) qc(nz) qi(nz) qs(nz) dz(nz) p(nz) t(nz) xland
! stdout file (arg 2): per column cldfra(nz) qc(nz) qi(nz) qv(nz)
program oracle
  use oracle_cldfra3, only: cal_cldfra3
  implicit none
  integer :: ncol, nz, c, u, v
  real :: gridkm, xland
  real, allocatable :: qv(:), qc(:), qi(:), qs(:), dz(:), p(:), t(:), cf(:)
  character(len=512) :: fin, fout
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  open(newunit=v, file=trim(fout), access='stream', form='unformatted', status='replace')
  read(u) ncol, nz, gridkm
  allocate(qv(nz), qc(nz), qi(nz), qs(nz), dz(nz), p(nz), t(nz), cf(nz))
  do c = 1, ncol
     read(u) qv, qc, qi, qs, dz, p, t, xland
     cf = 0.
     call cal_cldfra3(cf, qv, qc, qi, qs, dz, p, t, xland, gridkm, .false., 1.5, 1, nz, .false.)
     write(v) cf, qc, qi, qv
  end do
  close(u)
  close(v)
end program oracle

subroutine wrf_debug(level, str)
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_error_fatal(str)
  character(len=*), intent(in) :: str
  write(0, *) 'WRF_ERROR_FATAL: ', trim(str)
  stop 3
end subroutine wrf_error_fatal
