! =====================================================================
! v0.17 single-column Thompson graupel-hail (WRF mp_physics=38, THOMPSONGH)
! oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_mp_thompson.F scheme in its
! is_hail_aware (mp=38) configuration:
!   thompson_init  (called WITH `ng` PRESENT -> is_hail_aware=.TRUE.,
!                   dimNRHG=NRHG=9; this generates the variable-density
!                   graupel collision tables qr_acr_qg_mp38V1 in-memory --
!                   write_thompson_mp38table is forced .TRUE. by the stub
!                   nl_get so no .dat file is required).
!   mp_gt_driver   (the full Thompson microphysics 3D wrapper; called on a
!                   single column with qb=graupel-volume and ng=graupel-number
!                   PRESENT so the predicted-density graupel-hail hybrid path
!                   runs).
!
! mp=38 prognostic set (WRF Registry thompsongh):
!   moist : qv, qc, qr, qi, qs, qg
!   scalar: ni (qni), nr (qnr), nc (qnc), ng (qng), qb (qvolg),
!           nwfa (qnwfa), nifa (qnifa), nbca (qnbca)
! There is NO separate qh hail category: hail is represented as
! VARIABLE-DENSITY graupel via the predicted graupel volume qb (= the State
! leaf `qvolg`), with a per-cell density index into the rho_g(1:9) table.
!
! This is the GOLD oracle for the future JAX mp=38 port: it is the real
! Fortran Thompson scheme (module_mp_thompson.F), not a re-implementation, so
! the JAX port cannot "self-compare". The init is the real thompson_init so
! every saved module constant + lookup table (incl. the 9-plane variable-
! density collision tables) is exactly WRF.
!
! Usage: ./thompgh_oracle <case_id>   (1..6 regimes, see build_column)
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! Default REAL = REAL*4 (single precision); the JAX port runs fp64. Parity is
! therefore to a PREDECLARED physical tolerance, never bitwise (see
! run_thompgh_parity.py). A -DDOUBLE_PRECISION build (-fdefault-real-8)
! provides the fp64 reference used for the categorical effective-radius
! diagnostics.
! =====================================================================
PROGRAM thompgh_oracle
  USE module_mp_thompson, ONLY : thompson_init, mp_gt_driver, rslf, rsif
  IMPLICIT NONE

  ! ---- WRF model constants, bound exactly as the microphysics driver does ----
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: SVPT0  = 273.15
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP

  INTEGER, PARAMETER :: KX  = 40
  ! One ACTIVE interior column (i=j=1) but a 2-wide memory tile so the WRF
  ! init's `hgt(its+1, k, jts+1)` DEBUG-column probe (module_mp_thompson.F:476)
  ! and the `its:ite-1`/`jts:jte-1` aerosol MAXVAL slices stay in bounds. The
  ! compute domain is ide=jde=2 so i_end=MIN(ite,ide-1)=1, j_end=1: only the
  ! single column (1,*,1) is integrated by the scheme.
  INTEGER, PARAMETER :: KX1 = KX + 1
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX1
  INTEGER, PARAMETER :: ims=1, ime=2, jms=1, jme=2, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=2, jts=1, jte=2, kts=1, kte=KX

  ! single column (1,KX,1) prognostic arrays
  REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: &
        qv, qc, qr, qi, qs, qg, ni, nr, th, &
        nc, ng, qb, nwfa, nifa, nbca, &
        pii, p, w, dz, refl_10cm, re_cloud, re_ice, re_snow
  REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: hgt3d
  REAL, DIMENSION(ims:ime, jms:jme) :: &
        RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAUPELNC, GRAUPELNCV, SR
  REAL, DIMENSION(ims:ime, jms:jme) :: nwfa2d, nifa2d, nbca2d

  ! saved copies of the inputs for dumping
  REAL, DIMENSION(kts:kte) :: t0, qv0, qc0, qr0, qi0, qs0, qg0
  REAL, DIMENSION(kts:kte) :: ni0, nr0, nc0, ng0, qb0, nwfa0, nifa0, nbca0
  REAL, DIMENSION(kts:kte) :: pii0, p0, dz0, rho0

  REAL :: DT
  INTEGER :: variant, ivar, k, case_id, has_reqc, has_reqi, has_reqs
  CHARACTER(LEN=32) :: arg
  CHARACTER(LEN=64) :: input_path
  REAL :: direction
  REAL :: x_qv,x_qc,x_qr,x_qi,x_qs,x_qg,x_ni,x_nr,x_nc,x_ng,x_qb,x_nwfa,x_nifa,x_th,x_p,x_pii,x_w,x_dz,x_nbca

  ! ---- parse case id ----
  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 90.0   ! representative convection-permitting microphysics dt (s)

  ! ---- build the chosen column (sets qv,qc,...,qg, ni,nr,nc,ng,qb,
  !      nwfa,nifa,nbca, th, pii, p, w, dz, rho) ----
  IF (case_id <= 6) THEN
    WRITE(input_path,'(A,I0,A)') 'input',case_id,'.txt'
  OPEN(97,FILE=TRIM(input_path),STATUS='OLD')
  READ(97,*) DT
  DO k=kts,kte
    READ(97,*) x_qv,x_qc,x_qr,x_qi,x_qs,x_qg,x_ni,x_nr,x_nc,x_ng,x_qb,x_nwfa,x_nifa,x_th,x_p,x_pii,x_w,x_dz,x_nbca
    qv(:,k,:)=x_qv
    qc(:,k,:)=x_qc
    qr(:,k,:)=x_qr
    qi(:,k,:)=x_qi
    qs(:,k,:)=x_qs
    qg(:,k,:)=x_qg
    ni(:,k,:)=x_ni
    nr(:,k,:)=x_nr
    nc(:,k,:)=x_nc
    ng(:,k,:)=x_ng
    qb(:,k,:)=x_qb
    nwfa(:,k,:)=x_nwfa
    nifa(:,k,:)=x_nifa
    th(:,k,:)=x_th
    p(:,k,:)=x_p
    pii(:,k,:)=x_pii
    w(:,k,:)=x_w
    dz(:,k,:)=x_dz
    nbca(:,k,:)=x_nbca
  END DO
  CLOSE(97)

  ELSE
    CALL build_column(4)
    DO k=kts,kte
      IF (case_id == 7) THEN
        th(1,k,1)=230.0/pii(1,k,1)
        qv(1,k,1)=1.5*rsif(p(1,k,1),230.0)
        qc(1,k,1)=0.;qr(1,k,1)=0.;qi(1,k,1)=0.;qs(1,k,1)=0.;qg(1,k,1)=0.
        nc(1,k,1)=0.;nr(1,k,1)=0.;ni(1,k,1)=0.;ng(1,k,1)=0.;qb(1,k,1)=0.
        w(1,k,1)=1.
      ELSEIF(case_id == 8) THEN
        th(1,k,1)=268.0/pii(1,k,1)
        qv(1,k,1)=1.02*rsif(p(1,k,1),268.0)
        qc(1,k,1)=0.01;qr(1,k,1)=1.E-3;qi(1,k,1)=1.E-4;qs(1,k,1)=1.E-3;qg(1,k,1)=1.E-3
        nc(1,k,1)=2.E6;nr(1,k,1)=1.E4;ni(1,k,1)=1.E6;ng(1,k,1)=1.E4
        qb(1,k,1)=qg(1,k,1)/(50.+100.*real(mod(k-1,9)))
        w(1,k,1)=3.;nifa(1,k,1)=1.E7
      ELSEIF(case_id == 9) THEN
        th(1,k,1)=263.0/pii(1,k,1)
        qv(1,k,1)=1.01*rslf(p(1,k,1),263.0)
        qc(1,k,1)=1.E-4;qr(1,k,1)=1.E-3;qi(1,k,1)=1.E-5;qs(1,k,1)=0.;qg(1,k,1)=1.E-4
        nc(1,k,1)=100.E6;nr(1,k,1)=10.;ni(1,k,1)=1.E7;ng(1,k,1)=1.E3
        qb(1,k,1)=qg(1,k,1)/800.;w(1,k,1)=2.
      ENDIF
    ENDDO
    DT=18.
  ENDIF

  hgt3d = 0.0
  nwfa2d = 0.0; nifa2d = 0.0; nbca2d = 0.0

  ! Production physics_init.F:4565: BOTH aerosol and hail switches.
  CALL thompson_init(HGT=hgt3d, ORHO=(287.04*(th*pii)*(qv+0.622))/(0.622*p), &
                     NWFA2D=nwfa2d,NWFA=nwfa,NIFA=nifa,NBCA=nbca, &
                     NG=ng,wif_input_opt=1,is_start=.TRUE., &
                     FRC_URB2D=nwfa2d,DX=3000.,DY=3000., &
                     ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
                     ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
                     its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte)

  DO variant=0,0
  WRITE(input_path,'(A,I0,A)') 'input',case_id,'.txt'
  OPEN(97,FILE=TRIM(input_path),STATUS='OLD')
  READ(97,*) DT
  DO k=kts,kte
    READ(97,*) x_qv,x_qc,x_qr,x_qi,x_qs,x_qg,x_ni,x_nr,x_nc,x_ng,x_qb,x_nwfa,x_nifa,x_th,x_p,x_pii,x_w,x_dz,x_nbca
    qv(:,k,:)=x_qv
    qc(:,k,:)=x_qc
    qr(:,k,:)=x_qr
    qi(:,k,:)=x_qi
    qs(:,k,:)=x_qs
    qg(:,k,:)=x_qg
    ni(:,k,:)=x_ni
    nr(:,k,:)=x_nr
    nc(:,k,:)=x_nc
    ng(:,k,:)=x_ng
    qb(:,k,:)=x_qb
    nwfa(:,k,:)=x_nwfa
    nifa(:,k,:)=x_nifa
    th(:,k,:)=x_th
    p(:,k,:)=x_p
    pii(:,k,:)=x_pii
    w(:,k,:)=x_w
    dz(:,k,:)=x_dz
    nbca(:,k,:)=x_nbca
  END DO
  CLOSE(97)
  IF (variant>0) THEN
    ivar=(variant+1)/2
    direction=1.
    IF(MOD(variant,2)==1) direction=-1.
    SELECT CASE(ivar)
    CASE(1)
      WHERE(qv/=0.) qv=NEAREST(qv,direction)
    CASE(2)
      WHERE(qc/=0.) qc=NEAREST(qc,direction)
    CASE(3)
      WHERE(qr/=0.) qr=NEAREST(qr,direction)
    CASE(4)
      WHERE(qi/=0.) qi=NEAREST(qi,direction)
    CASE(5)
      WHERE(qs/=0.) qs=NEAREST(qs,direction)
    CASE(6)
      WHERE(qg/=0.) qg=NEAREST(qg,direction)
    CASE(7)
      WHERE(ni/=0.) ni=NEAREST(ni,direction)
    CASE(8)
      WHERE(nr/=0.) nr=NEAREST(nr,direction)
    CASE(9)
      WHERE(nc/=0.) nc=NEAREST(nc,direction)
    CASE(10)
      WHERE(ng/=0.) ng=NEAREST(ng,direction)
    CASE(11)
      WHERE(qb/=0.) qb=NEAREST(qb,direction)
    CASE(12)
      WHERE(nwfa/=0.) nwfa=NEAREST(nwfa,direction)
    CASE(13)
      WHERE(nifa/=0.) nifa=NEAREST(nifa,direction)
    CASE(14)
      WHERE(th/=0.) th=NEAREST(th,direction)
    CASE(15)
      WHERE(p/=0.) p=NEAREST(p,direction)
    CASE(16)
      WHERE(pii/=0.) pii=NEAREST(pii,direction)
    CASE(17)
      WHERE(w/=0.) w=NEAREST(w,direction)
    CASE(18)
      WHERE(dz/=0.) dz=NEAREST(dz,direction)
    END SELECT
  END IF
  WRITE(*,'(A,I0)') 'VARIANT=',variant
  ! save inputs (the EXACT pre-call values the scheme receives).
  DO k = kts, kte
    t0(k)   = th(1,k,1)*pii(1,k,1)
    qv0(k)  = qv(1,k,1)
    qc0(k)  = qc(1,k,1)
    qr0(k)  = qr(1,k,1)
    qi0(k)  = qi(1,k,1)
    qs0(k)  = qs(1,k,1)
    qg0(k)  = qg(1,k,1)
    ni0(k)  = ni(1,k,1)
    nr0(k)  = nr(1,k,1)
    nc0(k)  = nc(1,k,1)
    ng0(k)  = ng(1,k,1)
    qb0(k)  = qb(1,k,1)
    nwfa0(k)= nwfa(1,k,1)
    nifa0(k)= nifa(1,k,1)
    nbca0(k)= nbca(1,k,1)
    pii0(k) = pii(1,k,1)
    p0(k)   = p(1,k,1)
    dz0(k)  = dz(1,k,1)
    rho0(k) = p(1,k,1)/(R_D*t0(k)*(1.0+0.608*qv(1,k,1)))
  END DO

  ! accumulators start at 0; the ncv + sr are set inside the driver.
  RAINNC=0.; RAINNCV=0.; SR=0.
  SNOWNC=0.; SNOWNCV=0.; GRAUPELNC=0.; GRAUPELNCV=0.
  refl_10cm=0.
  re_cloud=0.; re_ice=0.; re_snow=0.
  has_reqc=0; has_reqi=0; has_reqs=0

  ! ---- call the real Thompson graupel-hail microphysics ----
  CALL mp_gt_driver(qv, qc, qr, qi, qs, qg, qb, ni, nr, nc, ng,        &
                    nwfa, nifa, nbca, nwfa2d, nifa2d, nbca2d,          &
                    aer_init_opt=0, wif_input_opt=1,                   &
                    th=th, pii=pii, p=p, w=w, dz=dz,                   &
                    dt_in=DT, itimestep=1,                             &
                    RAINNC=RAINNC, RAINNCV=RAINNCV,                    &
                    SNOWNC=SNOWNC, SNOWNCV=SNOWNCV,                    &
                    GRAUPELNC=GRAUPELNC, GRAUPELNCV=GRAUPELNCV, SR=SR, &
                    refl_10cm=refl_10cm, diagflag=.FALSE., ke_diag=1,  &
                    do_radar_ref=0,                                    &
                    re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
                    has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
                    ids=ids,ide=ide, jds=jds,jde=jde, kds=kds,kde=kde, &
                    ims=ims,ime=ime, jms=jms,jme=jme, kms=kms,kme=kme, &
                    its=its,ite=ite, jts=jts,jte=jte, kts=kts,kte=kte)

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  ! inputs
  CALL dump_col('T_IN',    t0)
  CALL dump_col('QV_IN',   qv0)
  CALL dump_col('QC_IN',   qc0)
  CALL dump_col('QR_IN',   qr0)
  CALL dump_col('QI_IN',   qi0)
  CALL dump_col('QS_IN',   qs0)
  CALL dump_col('QG_IN',   qg0)
  CALL dump_col('NI_IN',   ni0)
  CALL dump_col('NR_IN',   nr0)
  CALL dump_col('NC_IN',   nc0)
  CALL dump_col('NG_IN',   ng0)
  CALL dump_col('QB_IN',   qb0)
  CALL dump_col('NWFA_IN', nwfa0)
  CALL dump_col('NIFA_IN', nifa0)
  CALL dump_col('NBCA_IN', nbca0)
  CALL dump_col('PII',     pii0)
  CALL dump_col('P',       p0)
  CALL dump_col('W_IN', w(1,:,1))
  CALL dump_col('DELZ',    dz0)
  CALL dump_col('DEN',     rho0)
  ! outputs (post-Thompson-GH)
  CALL dump_3d('T_OUT',    th, .TRUE.)
  CALL dump_3d('TH_OUT', th, .FALSE.)
  CALL dump_3d('QV_OUT',   qv, .FALSE.)
  CALL dump_3d('QC_OUT',   qc, .FALSE.)
  CALL dump_3d('QR_OUT',   qr, .FALSE.)
  CALL dump_3d('QI_OUT',   qi, .FALSE.)
  CALL dump_3d('QS_OUT',   qs, .FALSE.)
  CALL dump_3d('QG_OUT',   qg, .FALSE.)
  CALL dump_3d('NI_OUT',   ni, .FALSE.)
  CALL dump_3d('NR_OUT',   nr, .FALSE.)
  CALL dump_3d('NC_OUT',   nc, .FALSE.)
  CALL dump_3d('NWFA_OUT', nwfa, .FALSE.)
  CALL dump_3d('NIFA_OUT', nifa, .FALSE.)
  CALL dump_3d('NBCA_OUT', nbca, .FALSE.)
  CALL dump_3d('NG_OUT',   ng, .FALSE.)
  CALL dump_3d('QB_OUT',   qb, .FALSE.)
  CALL dump_3d('RE_CLOUD', re_cloud, .FALSE.)
  CALL dump_3d('RE_ICE',   re_ice, .FALSE.)
  CALL dump_3d('RE_SNOW',  re_snow, .FALSE.)
  WRITE(*,'(A,ES23.15)') 'RAINNC=',     RAINNC(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    RAINNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNC=',     SNOWNC(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    SNOWNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNC=',  GRAUPELNC(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNCV=', GRAUPELNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',         SR(1,1)

  END DO

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  ! Dump a (1,KX,1) field. If is_theta, convert to T via pii for the T_OUT key.
  SUBROUTINE dump_3d(name, arr, is_theta)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    LOGICAL, INTENT(IN) :: is_theta
    INTEGER :: kk
    REAL :: val
    DO kk = kts, kte
      IF (is_theta) THEN
        val = arr(1,kk,1)*pii(1,kk,1)
      ELSE
        val = arr(1,kk,1)
      END IF
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', val
    END DO
  END SUBROUTINE dump_3d

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id (same regimes as the WSM7 oracle so
  ! the two hail families are exercised over comparable thermodynamics):
  !  1 = warm moist BL, supersaturated low levels
  !  2 = deep mixed-phase: warm below, ice/snow/graupel aloft, melting
  !  3 = cold ice/snow column (all subfreezing), supersaturated wrt ice
  !  4 = graupel-dominant convective core: large qr/qi/qg + updraft (HAIL)
  !  5 = subsaturated mid-level with rain/snow/graupel falling
  !  6 = clean/near-zero hydrometeor column with slight supersaturation
  ! Bottom-up index (k=1 lowest model layer) matching Thompson ordering.
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid)
    INTEGER, INTENT(IN) :: cid
    REAL :: psfc, tsfc, theta_sfc, ztop
    REAL :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL :: lapse, rh_ml, rh_trop, zml, rho_k, qg_k, qb_k
    REAL, DIMENSION(KX) :: zz
    INTEGER :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ( (REAL(kk)-0.5)/REAL(KX) )**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40
    CASE (2)
      psfc=1000.0E2; tsfc=287.0; zml=600.0;  lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60
    CASE (3)
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70
    CASE (4)
      psfc=1000.0E2; tsfc=296.0; zml=1000.0; lapse=6.5E-3; rh_ml=1.00; rh_trop=0.65
    CASE (5)
      psfc=950.0E2;  tsfc=283.0; zml=300.0;  lapse=6.0E-3; rh_ml=0.55; rh_trop=0.30
    CASE (6)
      psfc=1000.0E2; tsfc=295.0; zml=2000.0; lapse=5.0E-3; rh_ml=1.01; rh_trop=0.50
    CASE DEFAULT
      psfc=1000.0E2; tsfc=295.0; zml=1000.0; lapse=5.5E-3; rh_ml=0.95; rh_trop=0.50
    END SELECT

    ! zero everything first (whole arrays incl. the inactive memory column 2,
    ! so the init's debug/MAXVAL probes never read uninitialized memory).
    qv=0.; qc=0.; qr=0.; qi=0.; qs=0.; qg=0.
    ni=0.; nr=0.; nc=0.; ng=0.; qb=0.
    nwfa=0.; nifa=0.; nbca=0.; w=0.
    th=0.; pii=1.; p=0.; dz=1.

    theta_sfc = tsfc * (P1000/psfc)**ROVCP
    p_k = psfc
    DO kk = 1, KX
      z_k = zz(kk)
      IF (z_k <= zml) THEN
        th_k = theta_sfc
      ELSE
        th_k = theta_sfc + lapse*(z_k - zml)
      END IF
      IF (z_k <= zml) THEN
        rh_k = rh_ml
      ELSE
        rh_k = rh_trop + (rh_ml-rh_trop)*EXP(-(z_k-zml)/3000.0)
      END IF
      IF (kk == 1) THEN
        t_k  = th_k*(psfc/P1000)**ROVCP
        tv_k = t_k
        p_k  = psfc * EXP(-G*zz(1)/(R_D*tv_k))
      ELSE
        t_k  = th_k*(p_k/P1000)**ROVCP
        tv_k = t_k*(1.0+0.608*qv(1,kk-1,1))
        p_k  = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      es  = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qv(1,kk,1)  = MAX(rh_k*qsw, 1.0E-8)
      pii(1,kk,1) = (p_k/P1000)**ROVCP
      th(1,kk,1)  = t_k/pii(1,kk,1)
      p(1,kk,1)   = p_k
      tv_k = t_k*(1.0+0.608*qv(1,kk,1))
      rho_k = p_k/(R_D*tv_k)
      IF (kk == 1) THEN
        dz(1,kk,1) = 2.0*zz(1)
      ELSE
        dz(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
    END DO

    ! Seed hydrometeors + number/volume per regime. Graupel volume qb seeds a
    ! representative graupel density rho_g0 (=400, the mp=8/idx_bg1 default) so
    ! qb = qg/rho_g0; the scheme then evolves the per-cell density.
    DO kk = 1, KX
      z_k = zz(kk)
      t_k = th(1,kk,1)*pii(1,kk,1)
      tv_k = t_k*(1.0+0.608*qv(1,kk,1))
      rho_k = p(1,kk,1)/(R_D*tv_k)
      SELECT CASE (cid)
      CASE (1)
        IF (z_k < 3000.0)  qc(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qr(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)
        IF (z_k < 4000.0)  qc(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qr(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qi(1,kk,1) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qs(1,kk,1) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qg(1,kk,1) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
      CASE (3)
        qi(1,kk,1) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qs(1,kk,1) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qg(1,kk,1) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
      CASE (4)
        IF (z_k < 5000.0)  qc(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qr(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qi(1,kk,1) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qs(1,kk,1) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qg(1,kk,1) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
      CASE (5)
        IF (z_k < 6000.0)  qr(1,kk,1) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qs(1,kk,1) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qg(1,kk,1) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
      CASE (6)
        IF (z_k < 3000.0)  qc(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT

      ! Number concentrations: cloud Nc ~ 100/cc (continental), graupel Ng and
      ! rain Nr from representative intercepts; ice Ni small. These are the
      ! prognostic scalar inputs (kg^-1) the mp=38 scheme reads.
      qg_k = qg(1,kk,1)
      IF (qc(1,kk,1) > 1.0E-9) nc(1,kk,1) = 100.0E6 / rho_k
      IF (qr(1,kk,1) > 1.0E-9) nr(1,kk,1) = 1.0E4   / rho_k
      IF (qi(1,kk,1) > 1.0E-12) ni(1,kk,1) = 5.0E4  / rho_k
      IF (qg_k > 1.0E-12) THEN
        ng(1,kk,1) = 1.0E4 / rho_k
        ! qb = graupel volume mixing ratio: qg / rho_g0 with rho_g0=400 kg/m^3
        ! (the mp=8 default density idx_bg1=5) -> a physical starting density.
        qb_k = qg_k / 400.0
        qb(1,kk,1) = qb_k
      END IF
      ! aerosol-aware backgrounds (water/ice-friendly + black carbon CCN/IN)
      nwfa(1,kk,1) = 1.0E9   ! ~1e3 /cc water-friendly aerosol
      nifa(1,kk,1) = 1.0E6   ! ~1   /cc ice-friendly aerosol
      nbca(1,kk,1) = 1.0E6   ! ~1   /cc black carbon aerosol
    END DO
  END SUBROUTINE build_column

END PROGRAM thompgh_oracle
