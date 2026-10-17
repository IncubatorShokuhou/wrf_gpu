! Thin public ACMPBL caller mirroring module_pbl_driver.F:1592-1610 (lane t-calleraudit).
! One column per call: ims=ime=its=ite=1, kts=1, kte=nz (mass), kme=nz+1, ide=2 (itf=1).
! prof(k,f): 1 U 2 V 3 PP3D 4 DZ8W 5 TH3D 6 T3D 7 QV 8 QC 9 QI 10 RR3D (A-grid mass levels)
! sfc(f): 1 UST 2 HFX 3 QFX 4 TSK 5 PSFC 6 PBLH 7 REGIME 8 GZ1OZ0 9 WSPD 10 PSIM 11 MUT 12 RMOL
!         13 DTPBL 14 XTIME(itimestep) 15 EP1 16 G 17 ROVCP 18 RD 19 CPD
! outp(k,f): 1 RUBLTEN 2 RVBLTEN 3 RTHBLTEN 4 RQVBLTEN 5 RQCBLTEN 6 RQIBLTEN 7 EXCH_H 8 EXCH_M
! osfc(f): 1 PBLH 2 KPBL 3 REGIME 4 UST 5 RMOL
subroutine probe(nz, prof, sfc, outp, osfc) bind(C, name='probe')
  use iso_c_binding
  use module_bl_acm, only: ACMPBL
  implicit none
  integer(c_int), value :: nz
  real(c_float), intent(in) :: prof(nz, 10), sfc(19)
  real(c_float), intent(out) :: outp(nz + 1, 8), osfc(5)
  real :: u(1, nz + 1, 1), v(1, nz + 1, 1), pp(1, nz + 1, 1), dz(1, nz + 1, 1), th(1, nz + 1, 1), &
          t(1, nz + 1, 1), qv(1, nz + 1, 1), qc(1, nz + 1, 1), qi(1, nz + 1, 1), rr(1, nz + 1, 1)
  real :: ru(1, nz + 1, 1), rv(1, nz + 1, 1), rth(1, nz + 1, 1), rqv(1, nz + 1, 1), rqc(1, nz + 1, 1), &
          rqi(1, nz + 1, 1), exh(1, nz + 1, 1), exm(1, nz + 1, 1)
  real :: ust(1, 1), hfx(1, 1), qfx(1, 1), tsk(1, 1), psfc(1, 1), pblh(1, 1), regime(1, 1), &
          gz1oz0(1, 1), wspd(1, 1), psim(1, 1), mut(1, 1), rmol(1, 1)
  integer :: kpbl(1, 1), k
  u = 0.; v = 0.; pp = 0.; dz = 0.; th = 0.; t = 0.; qv = 0.; qc = 0.; qi = 0.; rr = 0.
  ru = 0.; rv = 0.; rth = 0.; rqv = 0.; rqc = 0.; rqi = 0.; exh = 0.; exm = 0.
  do k = 1, nz
    u(1, k, 1) = prof(k, 1); v(1, k, 1) = prof(k, 2); pp(1, k, 1) = prof(k, 3); dz(1, k, 1) = prof(k, 4)
    th(1, k, 1) = prof(k, 5); t(1, k, 1) = prof(k, 6); qv(1, k, 1) = prof(k, 7); qc(1, k, 1) = prof(k, 8)
    qi(1, k, 1) = prof(k, 9); rr(1, k, 1) = prof(k, 10)
  end do
  ust = sfc(1); hfx = sfc(2); qfx = sfc(3); tsk = sfc(4); psfc = sfc(5); pblh = sfc(6); regime = sfc(7)
  gz1oz0 = sfc(8); wspd = sfc(9); psim = sfc(10); mut = sfc(11); rmol = sfc(12); kpbl = 0
  call ACMPBL(XTIME=nint(sfc(14)), DTPBL=sfc(13), U3D=u, V3D=v, PP3D=pp, DZ8W=dz, TH3D=th, T3D=t, &
              QV3D=qv, QC3D=qc, QI3D=qi, RR3D=rr, UST=ust, HFX=hfx, QFX=qfx, TSK=tsk, PSFC=psfc, &
              EP1=sfc(15), G=sfc(16), ROVCP=sfc(17), RD=sfc(18), CPD=sfc(19), PBLH=pblh, KPBL2D=kpbl, &
              EXCH_H=exh, EXCH_M=exm, REGIME=regime, GZ1OZ0=gz1oz0, WSPD=wspd, PSIM=psim, MUT=mut, &
              RMOL=rmol, RUBLTEN=ru, RVBLTEN=rv, RTHBLTEN=rth, RQVBLTEN=rqv, RQCBLTEN=rqc, RQIBLTEN=rqi, &
              ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz + 1, ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz + 1, &
              its=1, ite=1, jts=1, jte=1, kts=1, kte=nz)
  outp(:, 1) = ru(1, :, 1); outp(:, 2) = rv(1, :, 1); outp(:, 3) = rth(1, :, 1); outp(:, 4) = rqv(1, :, 1)
  outp(:, 5) = rqc(1, :, 1); outp(:, 6) = rqi(1, :, 1); outp(:, 7) = exh(1, :, 1); outp(:, 8) = exm(1, :, 1)
  osfc = [pblh(1, 1), real(kpbl(1, 1)), regime(1, 1), ust(1, 1), rmol(1, 1)]
end subroutine probe

subroutine wrf_error_fatal(message)
  character(len=*), intent(in) :: message
  print *, 'wrf_error_fatal: ', trim(message)
  stop 1
end subroutine wrf_error_fatal
