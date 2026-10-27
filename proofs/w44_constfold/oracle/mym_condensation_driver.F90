! o5-n333 oracle driver: calls the PRISTINE WRF 4.7.1 mym_condensation (module_bl_mynnedmf.o of
! <USER_HOME>/src/wrf_pristine/WRF, gfortran 14.3, REAL4 kind_phys) column by column.
! Caller wiring = mynn_bl_driver :905-913 (thl1,sqw1,sqv1,sqc1,sqi1,sqs1,p1,ex1,tsq1,qsq1,cov1,sh1,el1,
! cloudpdf, qc_bl1,qi_bl1,cldfra_bl1, pblh,hfx, vt1,vq1,th1,sgm1, spp_pbl,pattern_spp_pbl1).
! vt1/vq1/sgm1 are driver locals zeroed only at initflag; they are fully overwritten for k<kte by every CASE.
! Input  (stream, little endian): int32 ncol,nz,cloudpdf ; per column REAL4: dx,xland,pblh,hfx,
!        dz(nz),zw(nz+1),p(nz),exner(nz),thl(nz),qw(nz),qv(nz),qc(nz),qi(nz),qs(nz),
!        tsq(nz),qsq(nz),cov(nz),sh(nz),el(nz),th(nz)
! Output (stream): per column REAL4 qc_bl(nz),qi_bl(nz),cldfra_bl(nz),sgm(nz),vt(nz),vq(nz)
program mym_condensation_driver
  use ccpp_kind_types, only: kind_phys
  use module_bl_mynnedmf, only: mym_condensation
  implicit none
  integer :: ncol, nz, cloudpdf, icol, spp
  real(kind_phys) :: dx, xland, pblh, hfx
  real(kind_phys), allocatable :: dz(:), zw(:), p(:), exner(:), thl(:), qw(:), qv(:), qc(:), qi(:), qs(:)
  real(kind_phys), allocatable :: tsq(:), qsq(:), cov(:), sh(:), el(:), th(:)
  real(kind_phys), allocatable :: qc_bl(:), qi_bl(:), cldfra_bl(:), sgm(:), vt(:), vq(:), pattern(:)
  character(len=1024) :: inpath, outpath
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(10, file=trim(inpath), access='stream', form='unformatted', status='old', action='read')
  read(10) ncol, nz, cloudpdf
  allocate(dz(nz), zw(nz+1), p(nz), exner(nz), thl(nz), qw(nz), qv(nz), qc(nz), qi(nz), qs(nz))
  allocate(tsq(nz), qsq(nz), cov(nz), sh(nz), el(nz), th(nz))
  allocate(qc_bl(nz), qi_bl(nz), cldfra_bl(nz), sgm(nz), vt(nz), vq(nz), pattern(nz))
  open(20, file=trim(outpath), access='stream', form='unformatted', status='replace', action='write')
  spp = 0
  pattern = 0.0
  do icol = 1, ncol
     read(10) dx, xland, pblh, hfx
     read(10) dz, zw, p, exner, thl, qw, qv, qc, qi, qs, tsq, qsq, cov, sh, el, th
     vt = 0.0; vq = 0.0; sgm = 0.0
     qc_bl = 0.0; qi_bl = 0.0; cldfra_bl = 0.0
     call mym_condensation(1, nz, dx, dz, zw, xland, thl, qw, qv, qc, qi, qs, p, exner, &
          tsq, qsq, cov, sh, el, cloudpdf, qc_bl, qi_bl, cldfra_bl, pblh, hfx, &
          vt, vq, th, sgm, spp, pattern)
     write(20) qc_bl, qi_bl, cldfra_bl, sgm, vt, vq
  end do
  close(10); close(20)
end program mym_condensation_driver
