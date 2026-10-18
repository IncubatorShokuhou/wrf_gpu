! Caller for verbatim WRF 4.7.1 cal_cldfra2 (module_radiation_driver.F:1446-1453 wiring:
! F_QC/F_QI from the microphysics species set). stdin layout: int32 n, int32 f_qc, int32 f_qi,
! then qc(n) qi(n); output file: cldfra(n).
program oracle2
  use oracle_cldfra2, only: cal_cldfra2
  implicit none
  integer :: n, fqc, fqi, u, v
  real, allocatable :: qc(:,:,:), qi(:,:,:), cf(:,:,:)
  character(len=512) :: fin, fout
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  open(newunit=v, file=trim(fout), access='stream', form='unformatted', status='replace')
  read(u) n, fqc, fqi
  allocate(qc(n,1,1), qi(n,1,1), cf(n,1,1))
  read(u) qc, qi
  cf = -1.
  call cal_cldfra2(cf, qc, qi, fqc /= 0, fqi /= 0, 1,n, 1,1, 1,1, 1,n, 1,1, 1,1, 1,n, 1,1, 1,1)
  write(v) cf
  close(u)
  close(v)
end program oracle2
