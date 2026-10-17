! t-wsmdz: run the UNMODIFIED WRF wsm5/wsm6 driver-level wrappers on N real
! columns read from a stream file. Arguments are bound exactly as in
! phys/module_microphysics_driver.F CALL wsm5 (:2556) / CALL wsm6 (:2599).
! Input  (stream): int32 scheme(4|6), ncol, nk; real32 dt;
!                  11 x real32(ncol,nk): th,qv,qc,qr,qi,qs,qg,den,pii,p,delz
! Output (stream): 7 x real32(ncol,nk): th,qv,qc,qr,qi,qs,qg ;
!                  4 x real32(ncol): rainncv,snowncv,graupelncv,sr
program wsm_cols
  use module_model_constants, only: G, R_D, CP, R_V, CPV, CLIQ, CICE, PSAT, &
       XLV, XLS, XLF, SVPT0, EP_1, EP_2, EPSILON, RHOAIR0, RHOWATER, RHOSNOW, &
       RE_QC_BG, RE_QI_BG, RE_QS_BG, RE_QC_MAX, RE_QI_MAX, RE_QS_MAX
  use module_mp_wsm5, only: wsm5, wsm5init
  use module_mp_wsm6, only: wsm6
  use mp_wsm6, only: mp_wsm6_init
  implicit none
  integer :: scheme, ncol, nk, errflg
  real :: dt
  character(len=512) :: fin, fout
  character(len=256) :: errmsg
  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qi, qs, qg, den, pii, p, delz
  real, allocatable, dimension(:,:,:) :: re_c, re_i, re_s, refl
  real, allocatable, dimension(:,:) :: rain, rainncv, snow, snowncv, graupel, graupelncv, sr

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old')
  read(10) scheme, ncol, nk, dt
  allocate(th(ncol,nk,1), qv(ncol,nk,1), qc(ncol,nk,1), qr(ncol,nk,1), qi(ncol,nk,1), &
           qs(ncol,nk,1), qg(ncol,nk,1), den(ncol,nk,1), pii(ncol,nk,1), p(ncol,nk,1), &
           delz(ncol,nk,1), re_c(ncol,nk,1), re_i(ncol,nk,1), re_s(ncol,nk,1), refl(ncol,nk,1))
  allocate(rain(ncol,1), rainncv(ncol,1), snow(ncol,1), snowncv(ncol,1), graupel(ncol,1), &
           graupelncv(ncol,1), sr(ncol,1))
  read(10) th, qv, qc, qr, qi, qs, qg, den, pii, p, delz
  close(10)
  rain = 0.; rainncv = 0.; snow = 0.; snowncv = 0.; graupel = 0.; graupelncv = 0.; sr = 0.
  re_c = RE_QC_BG; re_i = RE_QI_BG; re_s = RE_QS_BG; refl = -35.

  if (scheme == 4) then
    call wsm5init(RHOAIR0, RHOWATER, RHOSNOW, CLIQ, CPV, .false.)
    call wsm5(TH=th, Q=qv, QC=qc, QR=qr, QI=qi, QS=qs,                       &
         DEN=den, PII=pii, P=p, DELZ=delz,                                   &
         DELT=dt, G=G, CPD=CP, CPV=CPV, RD=R_D, RV=R_V, T0C=SVPT0,           &
         EP1=EP_1, EP2=EP_2, QMIN=EPSILON, XLS=XLS, XLV0=XLV, XLF0=XLF,      &
         DEN0=RHOAIR0, DENR=RHOWATER, CLIQ=CLIQ, CICE=CICE, PSAT=PSAT,       &
         RAIN=rain, RAINNCV=rainncv, SNOW=snow, SNOWNCV=snowncv, SR=sr,      &
         REFL_10CM=refl, diagflag=.false., do_radar_ref=0,                   &
         has_reqc=0, has_reqi=0, has_reqs=0,                                 &
         re_cloud=re_c, re_ice=re_i, re_snow=re_s,                           &
         IDS=1, IDE=ncol+1, JDS=1, JDE=2, KDS=1, KDE=nk+1,                   &
         IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=nk,                       &
         ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nk)
  else if (scheme == 6) then
    call mp_wsm6_init(RHOAIR0, RHOWATER, RHOSNOW, CLIQ, CPV, 0, errmsg, errflg)
    call wsm6(TH=th, Q=qv, QC=qc, QR=qr, QI=qi, QS=qs, QG=qg,                &
         DEN=den, PII=pii, P=p, DELZ=delz,                                   &
         DELT=dt, G=G, CPD=CP, CPV=CPV, RD=R_D, RV=R_V, T0C=SVPT0,           &
         EP1=EP_1, EP2=EP_2, QMIN=EPSILON, XLS=XLS, XLV0=XLV, XLF0=XLF,      &
         DEN0=RHOAIR0, DENR=RHOWATER, CLIQ=CLIQ, CICE=CICE, PSAT=PSAT,       &
         RAIN=rain, RAINNCV=rainncv, SNOW=snow, SNOWNCV=snowncv, SR=sr,      &
         REFL_10CM=refl, diagflag=.false., do_radar_ref=0,                   &
         GRAUPEL=graupel, GRAUPELNCV=graupelncv,                             &
         has_reqc=0, has_reqi=0, has_reqs=0,                                 &
         re_qc_bg=RE_QC_BG, re_qi_bg=RE_QI_BG, re_qs_bg=RE_QS_BG,            &
         re_qc_max=RE_QC_MAX, re_qi_max=RE_QI_MAX, re_qs_max=RE_QS_MAX,      &
         re_cloud=re_c, re_ice=re_i, re_snow=re_s,                           &
         errmsg=errmsg, errflg=errflg,                                       &
         IDS=1, IDE=ncol+1, JDS=1, JDE=2, KDS=1, KDE=nk+1,                   &
         IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=nk,                       &
         ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nk)
  else
    stop 'scheme must be 4 or 6'
  end if

  open(11, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(11) th, qv, qc, qr, qi, qs, qg, rainncv, snowncv, graupelncv, sr
  close(11)
end program wsm_cols
