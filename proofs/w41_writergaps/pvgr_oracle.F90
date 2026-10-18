PROGRAM pvgr_oracle
  ! w41-writergaps: proofs/v040/urban_bep_bem/bep_bem_oracle.F90 plus history dumps of the
  ! PV/green-roof producers (EP_PV/QGR/TGR/DRAINGR) and a constructed SMOIS_URB=0.3 (the
  ! green-roof soil seed QR_URB4D=SMOIS_URB(:,1,:), module_sf_urban.F:2939). Pristine modules unchanged.
  ! Optional env PVGR_TRV_OFFSET [K]: constructed green-roof soil vs roof contrast added to the
  ! urban_var_init TRV_URB4D (=TBLEND, so wall_gr's hfgr starts at ~0 otherwise); unset = unchanged.
  ! Optional env PVGR_SMOIS_URB: constructed SMOIS_URB (default 0.3), e.g. dry soil below the wilting point.
  ! WRF BEP/BEM via the verbatim Noah-MP caller; derived from the SLUCM harness.
  ! Horizontal 1x1 tile, eight constructed levels above real WN3 surface forcings.
  ! Registry map sizes follow module_check_a_mundo.F:3274-3303.
  USE module_sf_urban, ONLY : urban_param_init, urban_var_init, FRC_URB_TBL
  USE noahmp_urban_extract, ONLY : noahmp_urban
  USE module_model_constants, ONLY : rcp
  USE module_bep_bem_helper, ONLY : nurbm
  IMPLICIT NONE

  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=9
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=9
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=8
  INTEGER, PARAMETER :: NSOIL=4, NL=4, NUHI=15, ND=2, NZU=18, NWR=10
#ifdef WITH_BEM
  INTEGER, PARAMETER :: SCHEME=3, NBUI=15
#else
  INTEGER, PARAMETER :: SCHEME=2, NBUI=1
#endif
  INTEGER, PARAMETER :: ZRD=ND*NWR*NZU, ZWD=ZRD*NBUI, GD=ND*10
  INTEGER, PARAMETER :: ZD=ND*NZU*NBUI, ZDF=ND*NZU, BD=NZU*NBUI, WD=ZD
  INTEGER, PARAMETER :: GBD=ND*10*NBUI, FBD=ND*(NZU-1)*10*NBUI, ZGRD=ND*10*NZU
  INTEGER, PARAMETER :: NVMAX=64

  CHARACTER(LEN=512) :: label, rfile, rname
  CHARACTER(LEN=1024) :: hdr
  CHARACTER(LEN=64) :: trv_offset_text
  INTEGER :: trv_offset_status, smois_status
  REAL :: trv_offset, smois_seed
  CHARACTER(LEN=64) :: smois_text
  INTEGER :: nreg, nv, r, s, nsteps, k, julian, julyr
  REAL :: v(NVMAX)
  REAL :: DZR(NL), DZB(NL), DZG(NL)

  INTEGER :: IVGTYP(ims:ime,jms:jme), UTYPE_URB2D(ims:ime,jms:jme)
  REAL :: dt, frc_in, cosz, xlat0, omg, declin, fveg, t2mv, t2mb, q2mv, q2mb, psfc
  REAL :: base_tsk, base_hfx, base_qfx, base_lh, base_grdflx, base_albedo, base_emiss
  REAL :: base_qsfc, base_chs, base_chs2, base_cqs2, base_ust, base_znt
  REAL :: T2, Q2, TH2, U10, V10

  REAL, DIMENSION(ims:ime,jms:jme) :: COSZ_URB2D, XLAT_URB2D, SWDOWN, SWDDIR, SWDDIF, GLW, &
       RAINBL, ZNT, TSK, HFX, QFX, LH, GRDFLX, ALBEDO, EMISS, QSFC, TMN, &
       CMR_SFCDIF, CHR_SFCDIF, CMC_SFCDIF, CHC_SFCDIF, CMGR_SFCDIF, CHGR_SFCDIF, &
       TR_URB2D, TB_URB2D, TG_URB2D, TC_URB2D, QC_URB2D, UC_URB2D, &
       XXXR_URB2D, XXXB_URB2D, XXXG_URB2D, XXXC_URB2D, &
       SH_URB2D, LH_URB2D, G_URB2D, RN_URB2D, TS_URB2D, PSIM_URB2D, PSIH_URB2D, &
       U10_URB2D, V10_URB2D, GZ1OZ0_URB2D, AKMS_URB2D, TH2_URB2D, Q2_URB2D, UST_URB2D, &
       OMG_URB2D, CMCR_URB2D, TGR_URB2D, DRELR_URB2D, DRELB_URB2D, DRELG_URB2D, &
       FLXHUMR_URB2D, FLXHUMB_URB2D, FLXHUMG_URB2D, FRC_URB2D, CHS, CHS2, CQS2, &
       SF_AC_URB3D, LF_AC_URB3D, CM_AC_URB3D, SFVENT_URB3D, LFVENT_URB3D, EP_PV_URB3D, &
       QGR_URB3D, TGR_URB3D, DRAINGR_URB3D, LP_URB2D, LB_URB2D, HGT_URB2D, MH_URB2D, &
       STDH_URB2D, LF_URB2D_S, Z0_URB2D, VEGFRA, UST, XLONG, XLAT
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: T3D, QV3D, U_PHY, V_PHY, P8W3D, DZ8W, &
       TH_PHY, RHO, P_PHY, A_U_BEP, A_V_BEP, A_T_BEP, A_Q_BEP, A_E_BEP, B_U_BEP, B_V_BEP, &
       B_T_BEP, B_Q_BEP, B_E_BEP, DLG_BEP, DL_U_BEP, SF_BEP, VL_BEP
  REAL, DIMENSION(ims:ime,1:NL,jms:jme) :: TRL_URB3D, TBL_URB3D, TGL_URB3D, TGRL_URB3D, &
       SMR_URB3D, TSLB, SMOIS_URB
  REAL, DIMENSION(ims:ime,1:ZRD,jms:jme) :: TRB_URB4D
  REAL, DIMENSION(ims:ime,1:ZWD,jms:jme) :: TW1_URB4D, TW2_URB4D
  REAL, DIMENSION(ims:ime,1:GD,jms:jme) :: TGB_URB4D
  REAL, DIMENSION(ims:ime,1:BD,jms:jme) :: TLEV_URB3D, QLEV_URB3D
  REAL, DIMENSION(ims:ime,1:WD,jms:jme) :: TW1LEV_URB3D, TW2LEV_URB3D, SFWIN1_URB3D, SFWIN2_URB3D
  REAL, DIMENSION(ims:ime,1:GBD,jms:jme) :: TGLEV_URB3D
  REAL, DIMENSION(ims:ime,1:FBD,jms:jme) :: TFLEV_URB3D
  REAL, DIMENSION(ims:ime,1:ZD,jms:jme) :: SFW1_URB3D, SFW2_URB3D
  REAL, DIMENSION(ims:ime,1:ZDF,jms:jme) :: SFR_URB3D, T_PV_URB3D, DRAIN_URB4D, SFRV_URB3D, LFRV_URB3D, DGR_URB3D, LFR_URB3D
  REAL, DIMENSION(ims:ime,1:ND,jms:jme) :: SFG_URB3D, DG_URB3D, LFG_URB3D
  REAL, DIMENSION(ims:ime,1:ZGRD,jms:jme) :: TRV_URB4D, QR_URB4D
  REAL, DIMENSION(ims:ime,1:NUHI,jms:jme) :: HI_URB2D
  REAL, DIMENSION(ims:ime,1:4,jms:jme) :: LF_URB2D
  REAL :: gmt, height
  INTEGER :: julday

  CALL get_command_argument(1, label)
  CALL get_command_argument(2, rfile)
  WRITE(*,'(A,A)') 'PRECISION=', TRIM(label)
  WRITE(*,'(A,I0)') 'REAL_KIND=', KIND(dt)

  ! module_check_a_mundo.F:458-462, use_wudapt_lcz=0 (shared module state).
  nurbm=3
  DZR = 0.; DZB = 0.; DZG = 0.
  CALL urban_param_init(DZR, DZB, DZG, NSOIL, SCHEME, 0, .FALSE.)
  DO k = 1, NL
    WRITE(*,'(A,I0,A,ES24.16E3)') 'DZR[', k, ']=', DZR(k)
    WRITE(*,'(A,I0,A,ES24.16E3)') 'DZB[', k, ']=', DZB(k)
    WRITE(*,'(A,I0,A,ES24.16E3)') 'DZG[', k, ']=', DZG(k)
  END DO

  OPEN(10, FILE=TRIM(rfile), STATUS='OLD', ACTION='READ')
  READ(10,*) nreg, nv
  READ(10,'(A)') hdr
  WRITE(*,'(A,I0)') 'NREG=', nreg

  DO r = 1, nreg
    READ(10,'(A)') rname
    READ(10,*) v(1:nv)
    WRITE(*,'(A,I0,A,A)') 'REGIME_NAME[', r, ']=', TRIM(rname)
    CALL zero_all()
    dt = v(1); nsteps = NINT(v(2)); julian = NINT(v(3)); julyr = NINT(v(4)); frc_in = v(5)
    WRITE(*,'(A,I0,A,I0)') 'NSTEPS[',r,']=',nsteps
    xlat0 = v(6); cosz = v(7); omg = v(8); declin = v(9)
    T3D(1,1,1) = v(10); QV3D(1,1,1) = v(11); U_PHY(1,1,1) = v(12); V_PHY(1,1,1) = v(13)
    SWDOWN(1,1) = v(14); GLW(1,1) = v(15); P8W3D(1,1,1) = v(16); P8W3D(1,2,1) = v(17)
    RAINBL(1,1) = v(18); DZ8W(1,1,1) = v(19); base_znt = v(20)
    base_tsk = v(21); base_hfx = v(22); base_qfx = v(23); base_lh = v(24); base_grdflx = v(25)
    base_albedo = v(26); base_emiss = v(27); base_qsfc = v(28); base_chs = v(29)
    base_chs2 = v(30); base_cqs2 = v(31); base_ust = v(32)
    fveg = v(33); t2mv = v(34); t2mb = v(35); q2mv = v(36); q2mb = v(37); psfc = v(38)
    DO k = 1, NL
      TSLB(1,k,1) = v(38+k)
    END DO
    TMN(1,1) = v(43); VEGFRA(1,1) = v(44)
    T3D(1,2,1) = T3D(1,1,1); QV3D(1,2,1) = QV3D(1,1,1)
    ! Explicit constructed vertical extension of real WN3 lowest-level forcing.
    DO k=1,kme
      DZ8W(1,k,1)=v(19)
      U_PHY(1,k,1)=v(12)*(1.+0.1*(k-1))
      V_PHY(1,k,1)=v(13)*(1.+0.1*(k-1))
      P_PHY(1,k,1)=v(16)*EXP(-9.81*((k-0.5)*v(19))/(287.*v(10)))
      TH_PHY(1,k,1)=v(10)*(1.e5/P_PHY(1,k,1))**RCP + 0.3*(k-1)
      T3D(1,k,1)=TH_PHY(1,k,1)*(P_PHY(1,k,1)/1.e5)**RCP
      RHO(1,k,1)=P_PHY(1,k,1)/(287.*T3D(1,k,1))
      QV3D(1,k,1)=v(11)*EXP(-0.1*(k-1))
    END DO
    IF (r==6) THEN
      ! Explicit multilayer-canopy stress column: interfaces 0,3,8,15,24,...
      ! Original five WN3-derived regimes above are unchanged.
      height=0.
      DO k=1,kme
        DZ8W(1,k,1)=3.+2.*(k-1)
        P_PHY(1,k,1)=v(16)*EXP(-9.81*(height+DZ8W(1,k,1)/2.)/(287.*v(10)))
        TH_PHY(1,k,1)=v(10)*(1.e5/P_PHY(1,k,1))**RCP + 0.3*(k-1)
        T3D(1,k,1)=TH_PHY(1,k,1)*(P_PHY(1,k,1)/1.e5)**RCP
        RHO(1,k,1)=P_PHY(1,k,1)/(287.*T3D(1,k,1))
        height=height+DZ8W(1,k,1)
      END DO
    END IF
    SWDDIR=0.8*SWDOWN; SWDDIF=0.2*SWDOWN
    COSZ_URB2D = cosz; XLAT_URB2D = xlat0; OMG_URB2D = omg; XLAT = xlat0
    IVGTYP = 13
    FRC_URB2D = frc_in
    TSK = base_tsk
    CALL dumpin(r, 'dt', dt); CALL dumpin(r, 'frc_in', frc_in); CALL dumpin(r, 'omg', omg)
    CALL dumpin(r, 't3d1', T3D(1,1,1)); CALL dumpin(r, 'qv1', QV3D(1,1,1))
    CALL dumpin(r, 'u1', U_PHY(1,1,1)); CALL dumpin(r, 'v1', V_PHY(1,1,1))
    CALL dumpin(r, 'swdown', SWDOWN(1,1)); CALL dumpin(r, 'glw', GLW(1,1))
    CALL dumpin(r, 'p8w1', P8W3D(1,1,1)); CALL dumpin(r, 'p8w2', P8W3D(1,2,1))
    CALL dumpin(r, 'rainbl', RAINBL(1,1)); CALL dumpin(r, 'dz8w1', DZ8W(1,1,1))
    CALL dumpin(r, 'tsk', base_tsk); CALL dumpin(r, 'hfx', base_hfx); CALL dumpin(r, 'qfx', base_qfx)
    CALL dumpin(r, 'lh', base_lh); CALL dumpin(r, 'grdflx', base_grdflx)
    CALL dumpin(r, 'albedo', base_albedo); CALL dumpin(r, 'qsfc', base_qsfc)
    CALL dumpin(r, 'chs', base_chs); CALL dumpin(r, 'chs2', base_chs2); CALL dumpin(r, 'cqs2', base_cqs2)
    CALL dumpin(r, 'ust', base_ust); CALL dumpin(r, 'fveg', fveg); CALL dumpin(r, 't2mv', t2mv)
    CALL dumpin(r, 't2mb', t2mb); CALL dumpin(r, 'q2mv', q2mv); CALL dumpin(r, 'q2mb', q2mb)
    CALL dumpin(r, 'psfc', psfc)
    DO k = 1, NL
      CALL dumpin_k(r, 'tslb', k, TSLB(1,k,1))
    END DO
    CALL dumpin(r, 'julian', REAL(julian)); CALL dumpin(r, 'julyr', REAL(julyr))

    SMOIS_URB = 0.3
    CALL GET_ENVIRONMENT_VARIABLE('PVGR_SMOIS_URB', smois_text, STATUS=smois_status)
    IF (smois_status == 0) THEN
      READ(smois_text, *) smois_seed
      SMOIS_URB = smois_seed
    END IF
    CALL urban_var_init(13, TSK, TSLB, TMN, IVGTYP,                     &
         ims, ime, jms, jme, kms, kme, NSOIL,                            &
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,                     &
         .FALSE., SCHEME,                                                     &
         XXXR_URB2D, XXXB_URB2D, XXXG_URB2D, XXXC_URB2D,                 &
         TR_URB2D, TB_URB2D, TG_URB2D, TC_URB2D, QC_URB2D,               &
         TRL_URB3D, TBL_URB3D, TGL_URB3D,                                &
         SH_URB2D, LH_URB2D, G_URB2D, RN_URB2D, TS_URB2D,                &
         ND, ZRD, ZWD, GD, ZD, ZDF, BD, WD, GBD, FBD, ZGRD, NUHI, &
         TRB_URB4D, TW1_URB4D, TW2_URB4D, TGB_URB4D,                     &
         TLEV_URB3D, QLEV_URB3D, TW1LEV_URB3D, TW2LEV_URB3D,             &
         TGLEV_URB3D, TFLEV_URB3D, SF_AC_URB3D, LF_AC_URB3D, CM_AC_URB3D, &
         SFVENT_URB3D, LFVENT_URB3D, SFWIN1_URB3D, SFWIN2_URB3D,         &
         SFW1_URB3D, SFW2_URB3D, SFR_URB3D, SFG_URB3D,                   &
         EP_PV_URB3D, T_PV_URB3D, TRV_URB4D, QR_URB4D, QGR_URB3D, TGR_URB3D, &
         DRAIN_URB4D, DRAINGR_URB3D, SFRV_URB3D, LFRV_URB3D, DGR_URB3D, DG_URB3D, &
         LFR_URB3D, LFG_URB3D, SMOIS_URB,                                &
         LP_URB2D, HI_URB2D, LB_URB2D, HGT_URB2D, MH_URB2D, STDH_URB2D, LF_URB2D, &
         CMCR_URB2D, TGR_URB2D, TGRL_URB3D, SMR_URB3D,                   &
         DRELR_URB2D, DRELB_URB2D, DRELG_URB2D,                          &
         FLXHUMR_URB2D, FLXHUMB_URB2D, FLXHUMG_URB2D,                    &
         A_U_BEP, A_V_BEP, A_T_BEP, A_Q_BEP, A_E_BEP, B_U_BEP, B_V_BEP,   &
         B_T_BEP, B_Q_BEP, B_E_BEP, DLG_BEP, DL_U_BEP, SF_BEP, VL_BEP,     &
         FRC_URB2D, UTYPE_URB2D, 0)
    CALL dumpin(r, 'frc_urb', FRC_URB2D(1,1)); CALL dumpin(r, 'utype', REAL(UTYPE_URB2D(1,1)))
    CALL GET_ENVIRONMENT_VARIABLE('PVGR_TRV_OFFSET', trv_offset_text, STATUS=trv_offset_status)
    IF (trv_offset_status == 0) THEN
      READ(trv_offset_text, *) trv_offset
      TRV_URB4D = TRV_URB4D + trv_offset
    END IF
    CALL dumpstate(r, 0)
    DO k=1,kme
      CALL dk(r,0,'IN_U_',k,U_PHY(1,k,1));CALL dk(r,0,'IN_V_',k,V_PHY(1,k,1))
      CALL dk(r,0,'IN_TH_',k,TH_PHY(1,k,1));CALL dk(r,0,'IN_RHO_',k,RHO(1,k,1))
      CALL dk(r,0,'IN_P_',k,P_PHY(1,k,1));CALL dk(r,0,'IN_DZ_',k,DZ8W(1,k,1))
      CALL dk(r,0,'IN_QV_',k,QV3D(1,k,1))
    END DO
    DO k=1,nv
      CALL dk(r,0,'IN_VALUES_',k,v(k))
    END DO
    CALL d(r,0,'IN_FRACTION',FRC_URB2D(1,1))

    DO s = 1, nsteps
      TSK = base_tsk; HFX = base_hfx; QFX = base_qfx; LH = base_lh; GRDFLX = base_grdflx
      ALBEDO = base_albedo; EMISS = base_emiss; QSFC = base_qsfc; ZNT = base_znt
      CHS = base_chs; CHS2 = base_chs2; CQS2 = base_cqs2; UST = base_ust
      CALL noahmp_urban(SCHEME, NSOIL, IVGTYP, s,                                        &
           dt, COSZ_URB2D, XLAT_URB2D,                                              &
           T3D, QV3D, U_PHY, V_PHY, SWDOWN,                                         &
           SWDDIR, SWDDIF,                                                          &
           GLW, P8W3D, RAINBL, DZ8W, ZNT,                                           &
           TSK, HFX, QFX, LH, GRDFLX,                                               &
           ALBEDO, EMISS, QSFC,                                                     &
           ids, ide, jds, jde, kds, kde,                                            &
           ims, ime, jms, jme, kms, kme,                                            &
           its, ite, jts, jte, kts, kte,                                            &
           CMR_SFCDIF, CHR_SFCDIF, CMC_SFCDIF,                                      &
           CHC_SFCDIF, CMGR_SFCDIF, CHGR_SFCDIF,                                    &
           TR_URB2D, TB_URB2D, TG_URB2D,                                            &
           TC_URB2D, QC_URB2D, UC_URB2D,                                            &
           XXXR_URB2D, XXXB_URB2D, XXXG_URB2D, XXXC_URB2D,                          &
           TRL_URB3D, TBL_URB3D, TGL_URB3D,                                         &
           SH_URB2D, LH_URB2D, G_URB2D, RN_URB2D, TS_URB2D,                         &
           PSIM_URB2D, PSIH_URB2D, U10_URB2D, V10_URB2D,                            &
           GZ1OZ0_URB2D, AKMS_URB2D,                                                &
           TH2_URB2D, Q2_URB2D, UST_URB2D,                                          &
           declin, OMG_URB2D,                                                       &
           NL, NL, NL,                                                              &
           DZR, DZB, DZG,                                                           &
           CMCR_URB2D, TGR_URB2D, TGRL_URB3D, SMR_URB3D,                            &
           DRELR_URB2D, DRELB_URB2D, DRELG_URB2D,                                   &
           FLXHUMR_URB2D, FLXHUMB_URB2D, FLXHUMG_URB2D,                             &
           julian, julyr,                                                           &
           FRC_URB2D, UTYPE_URB2D,                                                  &
           CHS, CHS2, CQS2,                                                         &
           ND, ZRD, ZWD, GD,                                                  &
           ZD, ZDF, BD, WD,                                                  &
           GBD, FBD, ZGRD,                                                        &
           NUHI,                                                                    &
           TRB_URB4D, TW1_URB4D, TW2_URB4D, TGB_URB4D,                              &
           TLEV_URB3D, QLEV_URB3D,                                                  &
           TW1LEV_URB3D, TW2LEV_URB3D,                                              &
           TGLEV_URB3D, TFLEV_URB3D,                                                &
           SF_AC_URB3D, LF_AC_URB3D, CM_AC_URB3D,                                   &
           SFVENT_URB3D, LFVENT_URB3D,                                              &
           SFWIN1_URB3D, SFWIN2_URB3D,                                              &
           SFW1_URB3D, SFW2_URB3D, SFR_URB3D, SFG_URB3D,                            &
           EP_PV_URB3D, T_PV_URB3D,                                                 &
           TRV_URB4D, QR_URB4D, QGR_URB3D, TGR_URB3D,                               &
           DRAIN_URB4D, DRAINGR_URB3D, SFRV_URB3D, LFRV_URB3D,                      &
           DGR_URB3D, DG_URB3D, LFR_URB3D, LFG_URB3D,                               &
           LP_URB2D, HI_URB2D, LB_URB2D, HGT_URB2D,                                 &
           MH_URB2D, STDH_URB2D, LF_URB2D,                                          &
           LF_URB2D_S, Z0_URB2D, VEGFRA,                                            &
           TH_PHY, RHO, P_PHY, UST,                                                 &
           gmt, julday, XLONG, XLAT,                                                &
           A_U_BEP, A_V_BEP, A_T_BEP, A_Q_BEP,                                      &
           A_E_BEP, B_U_BEP, B_V_BEP,                                               &
           B_T_BEP, B_Q_BEP, B_E_BEP, DLG_BEP,                                      &
           DL_U_BEP, SF_BEP, VL_BEP)
      ! module_surface_driver.F:3516-3520 (BEP/BEM urban overwrite).
      T2=TH_PHY(1,1,1)/((1.e5/psfc)**RCP)
      TH2=TH_PHY(1,1,1); Q2=QV3D(1,1,1)
      U10=U_PHY(1,1,1); V10=V_PHY(1,1,1)
      CALL dumpstate(r, s)
      CALL d(r, s, 'TSK', TSK(1,1)); CALL d(r, s, 'HFX', HFX(1,1)); CALL d(r, s, 'QFX', QFX(1,1))
      CALL d(r, s, 'LH', LH(1,1)); CALL d(r, s, 'GRDFLX', GRDFLX(1,1)); CALL d(r, s, 'ALBEDO', ALBEDO(1,1))
      CALL d(r, s, 'QSFC', QSFC(1,1)); CALL d(r, s, 'UST', UST(1,1)); CALL d(r, s, 'EMISS', EMISS(1,1))
      CALL d(r, s, 'ZNT', ZNT(1,1)); CALL d(r, s, 'CHS', CHS(1,1)); CALL d(r, s, 'CHS2', CHS2(1,1))
      CALL d(r, s, 'CQS2', CQS2(1,1))
      CALL d(r, s, 'TS_URB', TS_URB2D(1,1)); CALL d(r, s, 'SH_URB', SH_URB2D(1,1))
      CALL d(r, s, 'LH_URB', LH_URB2D(1,1)); CALL d(r, s, 'G_URB', G_URB2D(1,1))
      CALL d(r, s, 'RN_URB', RN_URB2D(1,1)); CALL d(r, s, 'PSIM_URB', PSIM_URB2D(1,1))
      CALL d(r, s, 'PSIH_URB', PSIH_URB2D(1,1)); CALL d(r, s, 'GZ1OZ0_URB', GZ1OZ0_URB2D(1,1))
      CALL d(r, s, 'U10_URB', U10_URB2D(1,1)); CALL d(r, s, 'V10_URB', V10_URB2D(1,1))
      CALL d(r, s, 'TH2_URB', TH2_URB2D(1,1)); CALL d(r, s, 'Q2_URB', Q2_URB2D(1,1))
      CALL d(r, s, 'UST_URB', UST_URB2D(1,1)); CALL d(r, s, 'AKMS_URB', AKMS_URB2D(1,1))
      CALL d(r, s, 'T2', T2); CALL d(r, s, 'Q2', Q2); CALL d(r, s, 'TH2', TH2)
      CALL d(r, s, 'U10', U10); CALL d(r, s, 'V10', V10)
    END DO
  END DO
  CLOSE(10)

CONTAINS

  SUBROUTINE zero_all()
    COSZ_URB2D=0.; XLAT_URB2D=0.; SWDOWN=0.; SWDDIR=0.; SWDDIF=0.; GLW=0.; RAINBL=0.; ZNT=0.
    TSK=0.; HFX=0.; QFX=0.; LH=0.; GRDFLX=0.; ALBEDO=0.; EMISS=0.; QSFC=0.; TMN=0.
    CMR_SFCDIF=0.; CHR_SFCDIF=0.; CMC_SFCDIF=0.; CHC_SFCDIF=0.; CMGR_SFCDIF=0.; CHGR_SFCDIF=0.
    TR_URB2D=0.; TB_URB2D=0.; TG_URB2D=0.; TC_URB2D=0.; QC_URB2D=0.; UC_URB2D=0.
    XXXR_URB2D=0.; XXXB_URB2D=0.; XXXG_URB2D=0.; XXXC_URB2D=0.
    SH_URB2D=0.; LH_URB2D=0.; G_URB2D=0.; RN_URB2D=0.; TS_URB2D=0.; PSIM_URB2D=0.; PSIH_URB2D=0.
    U10_URB2D=0.; V10_URB2D=0.; GZ1OZ0_URB2D=0.; AKMS_URB2D=0.; TH2_URB2D=0.; Q2_URB2D=0.
    UST_URB2D=0.; OMG_URB2D=0.; CMCR_URB2D=0.; TGR_URB2D=0.; DRELR_URB2D=0.; DRELB_URB2D=0.
    DRELG_URB2D=0.; FLXHUMR_URB2D=0.; FLXHUMB_URB2D=0.; FLXHUMG_URB2D=0.; FRC_URB2D=0.
    CHS=0.; CHS2=0.; CQS2=0.; SF_AC_URB3D=0.; LF_AC_URB3D=0.; CM_AC_URB3D=0.; SFVENT_URB3D=0.
    LFVENT_URB3D=0.; EP_PV_URB3D=0.; QGR_URB3D=0.; TGR_URB3D=0.; DRAINGR_URB3D=0.; LP_URB2D=0.
    LB_URB2D=0.; HGT_URB2D=0.; MH_URB2D=0.; STDH_URB2D=0.; LF_URB2D_S=0.; Z0_URB2D=0.; VEGFRA=0.
    UST=0.; XLONG=0.; XLAT=0.
    T3D=0.; QV3D=0.; U_PHY=0.; V_PHY=0.; P8W3D=0.; DZ8W=0.; TH_PHY=0.; RHO=0.; P_PHY=0.
    A_U_BEP=0.; A_V_BEP=0.; A_T_BEP=0.; A_Q_BEP=0.; A_E_BEP=0.; B_U_BEP=0.; B_V_BEP=0.
    B_T_BEP=0.; B_Q_BEP=0.; B_E_BEP=0.; DLG_BEP=0.; DL_U_BEP=0.; SF_BEP=0.; VL_BEP=0.
    TRL_URB3D=0.; TBL_URB3D=0.; TGL_URB3D=0.; TGRL_URB3D=0.; SMR_URB3D=0.; TSLB=0.; SMOIS_URB=0.
    TRB_URB4D=0.; TW1_URB4D=0.; TW2_URB4D=0.; TGB_URB4D=0.; TLEV_URB3D=0.; QLEV_URB3D=0.
    TW1LEV_URB3D=0.; TW2LEV_URB3D=0.; TGLEV_URB3D=0.; TFLEV_URB3D=0.; SFWIN1_URB3D=0.
    SFWIN2_URB3D=0.; SFW1_URB3D=0.; SFW2_URB3D=0.; SFR_URB3D=0.; SFG_URB3D=0.; T_PV_URB3D=0.
    TRV_URB4D=0.; QR_URB4D=0.; DRAIN_URB4D=0.; SFRV_URB3D=0.; LFRV_URB3D=0.; DGR_URB3D=0.
    DG_URB3D=0.; LFR_URB3D=0.; LFG_URB3D=0.; HI_URB2D=0.; LF_URB2D=0.
    UTYPE_URB2D=0; IVGTYP=0; gmt=0.; julday=1
  END SUBROUTINE zero_all

  SUBROUTINE dumpstate(rr, ss)
    INTEGER, INTENT(IN) :: rr, ss
    INTEGER :: kk
    DO kk=1,ZRD
      CALL dk(rr,ss,'TRB_URB4D_',kk,TRB_URB4D(1,kk,1))
    END DO
    DO kk=1,ZWD
      CALL dk(rr,ss,'TW1_URB4D_',kk,TW1_URB4D(1,kk,1))
    END DO
    DO kk=1,ZWD
      CALL dk(rr,ss,'TW2_URB4D_',kk,TW2_URB4D(1,kk,1))
    END DO
    DO kk=1,GD
      CALL dk(rr,ss,'TGB_URB4D_',kk,TGB_URB4D(1,kk,1))
    END DO
    DO kk=1,BD
      CALL dk(rr,ss,'TLEV_URB3D_',kk,TLEV_URB3D(1,kk,1))
    END DO
    DO kk=1,BD
      CALL dk(rr,ss,'QLEV_URB3D_',kk,QLEV_URB3D(1,kk,1))
    END DO
    DO kk=1,WD
      CALL dk(rr,ss,'TW1LEV_URB3D_',kk,TW1LEV_URB3D(1,kk,1))
    END DO
    DO kk=1,WD
      CALL dk(rr,ss,'TW2LEV_URB3D_',kk,TW2LEV_URB3D(1,kk,1))
    END DO
    DO kk=1,WD
      CALL dk(rr,ss,'SFWIN1_URB3D_',kk,SFWIN1_URB3D(1,kk,1))
    END DO
    DO kk=1,WD
      CALL dk(rr,ss,'SFWIN2_URB3D_',kk,SFWIN2_URB3D(1,kk,1))
    END DO
    DO kk=1,GBD
      CALL dk(rr,ss,'TGLEV_URB3D_',kk,TGLEV_URB3D(1,kk,1))
    END DO
    DO kk=1,FBD
      CALL dk(rr,ss,'TFLEV_URB3D_',kk,TFLEV_URB3D(1,kk,1))
    END DO
    DO kk=1,ZD
      CALL dk(rr,ss,'SFW1_URB3D_',kk,SFW1_URB3D(1,kk,1))
    END DO
    DO kk=1,ZD
      CALL dk(rr,ss,'SFW2_URB3D_',kk,SFW2_URB3D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'SFR_URB3D_',kk,SFR_URB3D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'T_PV_URB3D_',kk,T_PV_URB3D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'DRAIN_URB4D_',kk,DRAIN_URB4D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'SFRV_URB3D_',kk,SFRV_URB3D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'LFRV_URB3D_',kk,LFRV_URB3D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'DGR_URB3D_',kk,DGR_URB3D(1,kk,1))
    END DO
    DO kk=1,ZDF
      CALL dk(rr,ss,'LFR_URB3D_',kk,LFR_URB3D(1,kk,1))
    END DO
    DO kk=1,ND
      CALL dk(rr,ss,'SFG_URB3D_',kk,SFG_URB3D(1,kk,1))
    END DO
    DO kk=1,ND
      CALL dk(rr,ss,'DG_URB3D_',kk,DG_URB3D(1,kk,1))
    END DO
    DO kk=1,ND
      CALL dk(rr,ss,'LFG_URB3D_',kk,LFG_URB3D(1,kk,1))
    END DO
    DO kk=1,ZGRD
      CALL dk(rr,ss,'TRV_URB4D_',kk,TRV_URB4D(1,kk,1))
    END DO
    DO kk=1,ZGRD
      CALL dk(rr,ss,'QR_URB4D_',kk,QR_URB4D(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'A_U_BEP_',kk,A_U_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'A_V_BEP_',kk,A_V_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'A_T_BEP_',kk,A_T_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'A_Q_BEP_',kk,A_Q_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'A_E_BEP_',kk,A_E_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'B_U_BEP_',kk,B_U_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'B_V_BEP_',kk,B_V_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'B_T_BEP_',kk,B_T_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'B_Q_BEP_',kk,B_Q_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'B_E_BEP_',kk,B_E_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'DLG_BEP_',kk,DLG_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'DL_U_BEP_',kk,DL_U_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'SF_BEP_',kk,SF_BEP(1,kk,1))
    END DO
    DO kk=1,kme
      CALL dk(rr,ss,'VL_BEP_',kk,VL_BEP(1,kk,1))
    END DO
    CALL d(rr,ss,'SF_AC_URB3D',SF_AC_URB3D(1,1))
    CALL d(rr,ss,'LF_AC_URB3D',LF_AC_URB3D(1,1))
    CALL d(rr,ss,'CM_AC_URB3D',CM_AC_URB3D(1,1))
    CALL d(rr,ss,'SFVENT_URB3D',SFVENT_URB3D(1,1))
    CALL d(rr,ss,'LFVENT_URB3D',LFVENT_URB3D(1,1))
    CALL d(rr,ss,'EP_PV_URB3D',EP_PV_URB3D(1,1))
    CALL d(rr,ss,'QGR_URB3D',QGR_URB3D(1,1))
    CALL d(rr,ss,'TGR_URB3D',TGR_URB3D(1,1))
    CALL d(rr,ss,'DRAINGR_URB3D',DRAINGR_URB3D(1,1))
  END SUBROUTINE dumpstate

  SUBROUTINE d(rr, ss, nm, val)
    INTEGER, INTENT(IN) :: rr, ss
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, INTENT(IN) :: val
    WRITE(*,'(A,A,I0,A,I0,A,ES24.16E3)') nm, '[', rr, '][', ss, ']=', val
  END SUBROUTINE d

  SUBROUTINE dk(rr, ss, nm, kk, val)
    INTEGER, INTENT(IN) :: rr, ss, kk
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, INTENT(IN) :: val
    WRITE(*,'(A,I0,A,A,I0,A,I0,A,ES24.16E3)') nm, kk, '', '[', rr, '][', ss, ']=', val
  END SUBROUTINE dk

  SUBROUTINE dumpin(rr, nm, val)
    INTEGER, INTENT(IN) :: rr
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, INTENT(IN) :: val
    WRITE(*,'(A,A,A,I0,A,ES24.16E3)') 'IN_', nm, '[', rr, ']=', val
  END SUBROUTINE dumpin

  SUBROUTINE dumpin_k(rr, nm, kk, val)
    INTEGER, INTENT(IN) :: rr, kk
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, INTENT(IN) :: val
    WRITE(*,'(A,A,I0,A,I0,A,ES24.16E3)') 'IN_', nm, kk, '[', rr, ']=', val
  END SUBROUTINE dumpin_k

END PROGRAM pvgr_oracle
