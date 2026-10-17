PROGRAM urban_oracle
  !---------------------------------------------------------------------------
  ! v0.40 pristine-WRF oracle for the single-layer urban canopy model
  ! (sf_urban_physics=1) as coupled to Noah-MP.
  !
  ! * urban_param_init / urban_var_init / urban : UNMODIFIED phys/module_sf_urban.F
  ! * noahmp_urban : VERBATIM extract of phys/module_sf_noahmpdrv.F (lines
  !   2994-3783, SHA-checked by the build script) inside a wrapper module, so
  !   the per-cell caller wiring (forcing derivation, CHS/CHS2/CQS2 limits,
  !   FRC_URB2D blend, UST/QSFC/GRDFLX conventions) is WRF's own (E11).
  ! * the module_surface_driver.F urban T2/Q2/TH2/U10/V10 overwrite
  !   (:3485-3500) is reproduced below on a 1x1 tile.
  !
  ! Every regime is a 1x1 column; NSTEPS consecutive calls with the regime's
  ! forcing held fixed (the rural Noah-MP fluxes and the surface-layer
  ! CHS/CHS2/CQS2/UST are re-supplied every call, as the LSM/sfclay recompute
  ! them each step in WRF).  The full urban state and all outputs are dumped
  ! after every call.
  !
  ! Dump format: NAME[r]=v (inputs/init) and NAME[r][s]=v (after call s),
  ! reals ES24.16E3.  Arg1 = precision label, arg2 = regimes file.
  !---------------------------------------------------------------------------
  USE module_sf_urban, ONLY : urban_param_init, urban_var_init, FRC_URB_TBL
  USE noahmp_urban_extract, ONLY : noahmp_urban
  USE module_model_constants, ONLY : rcp
  IMPLICIT NONE

  INTEGER, PARAMETER :: ims=1, ime=2, jms=1, jme=1, kms=1, kme=2
  INTEGER, PARAMETER :: ids=1, ide=3, jds=1, jde=2, kds=1, kde=2
  INTEGER, PARAMETER :: its=1, ite=2, jts=1, jte=1, kts=1, kte=1
  INTEGER, PARAMETER :: NSOIL=4, NL=4, NUHI=1, NMAP=1
  INTEGER, PARAMETER :: NVMAX=64

  CHARACTER(LEN=512) :: label, rfile, rname
  CHARACTER(LEN=1024) :: hdr
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
  REAL, DIMENSION(ims:ime,1:NMAP,jms:jme) :: TRB_URB4D, TW1_URB4D, TW2_URB4D, TGB_URB4D, &
       TLEV_URB3D, QLEV_URB3D, TW1LEV_URB3D, TW2LEV_URB3D, TGLEV_URB3D, TFLEV_URB3D, &
       SFWIN1_URB3D, SFWIN2_URB3D, SFW1_URB3D, SFW2_URB3D, SFR_URB3D, SFG_URB3D, &
       T_PV_URB3D, TRV_URB4D, QR_URB4D, DRAIN_URB4D, SFRV_URB3D, LFRV_URB3D, DGR_URB3D, &
       DG_URB3D, LFR_URB3D, LFG_URB3D
  REAL, DIMENSION(ims:ime,1:NUHI,jms:jme) :: HI_URB2D
  REAL, DIMENSION(ims:ime,1:4,jms:jme) :: LF_URB2D
  REAL :: gmt
  INTEGER :: julday

  CALL get_command_argument(1, label)
  CALL get_command_argument(2, rfile)
  WRITE(*,'(A,A)') 'PRECISION=', TRIM(label)
  WRITE(*,'(A,I0)') 'REAL_KIND=', KIND(dt)

  DZR = 0.; DZB = 0.; DZG = 0.
  CALL urban_param_init(DZR, DZB, DZG, NSOIL, 1, 1, .FALSE.)
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
    COSZ_URB2D = cosz; XLAT_URB2D = xlat0; OMG_URB2D = omg; XLAT = xlat0
    IVGTYP = 51 + MOD(r-1,12)
    IF (IVGTYP(1,1)==62) IVGTYP=13
    FRC_URB2D = frc_in
    DZ8W(1,1,1) = 200.
    IF (MOD(r,2)==0) THEN
      MH_URB2D = 12.; HGT_URB2D = 12.; STDH_URB2D = 3.
      LP_URB2D = 0.3; LB_URB2D = 1.0
      IF (MOD(r,4)==0) LB_URB2D=0.2
      LF_URB2D(1,:,1) = (/ 0.1, 0.1, 0.1, 0.1 /)
    END IF
    CALL dumpin(r,'ivgtyp',REAL(IVGTYP(1,1)))
    CALL dumpin(r,'mh',MH_URB2D(1,1)); CALL dumpin(r,'hgt',HGT_URB2D(1,1))
    CALL dumpin(r,'stdh',STDH_URB2D(1,1)); CALL dumpin(r,'lp',LP_URB2D(1,1))
    CALL dumpin(r,'lb',LB_URB2D(1,1))
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

      IVGTYP(2,:)=IVGTYP(1,:)
      UTYPE_URB2D(2,:)=UTYPE_URB2D(1,:)
      COSZ_URB2D(2,:)=COSZ_URB2D(1,:)
      XLAT_URB2D(2,:)=XLAT_URB2D(1,:)
      SWDOWN(2,:)=SWDOWN(1,:)
      SWDDIR(2,:)=SWDDIR(1,:)
      SWDDIF(2,:)=SWDDIF(1,:)
      GLW(2,:)=GLW(1,:)
      RAINBL(2,:)=RAINBL(1,:)
      ZNT(2,:)=ZNT(1,:)
      TSK(2,:)=TSK(1,:)
      HFX(2,:)=HFX(1,:)
      QFX(2,:)=QFX(1,:)
      LH(2,:)=LH(1,:)
      GRDFLX(2,:)=GRDFLX(1,:)
      ALBEDO(2,:)=ALBEDO(1,:)
      EMISS(2,:)=EMISS(1,:)
      QSFC(2,:)=QSFC(1,:)
      TMN(2,:)=TMN(1,:)
      CMR_SFCDIF(2,:)=CMR_SFCDIF(1,:)
      CHR_SFCDIF(2,:)=CHR_SFCDIF(1,:)
      CMC_SFCDIF(2,:)=CMC_SFCDIF(1,:)
      CHC_SFCDIF(2,:)=CHC_SFCDIF(1,:)
      CMGR_SFCDIF(2,:)=CMGR_SFCDIF(1,:)
      CHGR_SFCDIF(2,:)=CHGR_SFCDIF(1,:)
      TR_URB2D(2,:)=TR_URB2D(1,:)
      TB_URB2D(2,:)=TB_URB2D(1,:)
      TG_URB2D(2,:)=TG_URB2D(1,:)
      TC_URB2D(2,:)=TC_URB2D(1,:)
      QC_URB2D(2,:)=QC_URB2D(1,:)
      UC_URB2D(2,:)=UC_URB2D(1,:)
      XXXR_URB2D(2,:)=XXXR_URB2D(1,:)
      XXXB_URB2D(2,:)=XXXB_URB2D(1,:)
      XXXG_URB2D(2,:)=XXXG_URB2D(1,:)
      XXXC_URB2D(2,:)=XXXC_URB2D(1,:)
      SH_URB2D(2,:)=SH_URB2D(1,:)
      LH_URB2D(2,:)=LH_URB2D(1,:)
      G_URB2D(2,:)=G_URB2D(1,:)
      RN_URB2D(2,:)=RN_URB2D(1,:)
      TS_URB2D(2,:)=TS_URB2D(1,:)
      PSIM_URB2D(2,:)=PSIM_URB2D(1,:)
      PSIH_URB2D(2,:)=PSIH_URB2D(1,:)
      U10_URB2D(2,:)=U10_URB2D(1,:)
      V10_URB2D(2,:)=V10_URB2D(1,:)
      GZ1OZ0_URB2D(2,:)=GZ1OZ0_URB2D(1,:)
      AKMS_URB2D(2,:)=AKMS_URB2D(1,:)
      TH2_URB2D(2,:)=TH2_URB2D(1,:)
      Q2_URB2D(2,:)=Q2_URB2D(1,:)
      UST_URB2D(2,:)=UST_URB2D(1,:)
      OMG_URB2D(2,:)=OMG_URB2D(1,:)
      CMCR_URB2D(2,:)=CMCR_URB2D(1,:)
      TGR_URB2D(2,:)=TGR_URB2D(1,:)
      DRELR_URB2D(2,:)=DRELR_URB2D(1,:)
      DRELB_URB2D(2,:)=DRELB_URB2D(1,:)
      DRELG_URB2D(2,:)=DRELG_URB2D(1,:)
      FLXHUMR_URB2D(2,:)=FLXHUMR_URB2D(1,:)
      FLXHUMB_URB2D(2,:)=FLXHUMB_URB2D(1,:)
      FLXHUMG_URB2D(2,:)=FLXHUMG_URB2D(1,:)
      FRC_URB2D(2,:)=FRC_URB2D(1,:)
      CHS(2,:)=CHS(1,:)
      CHS2(2,:)=CHS2(1,:)
      CQS2(2,:)=CQS2(1,:)
      SF_AC_URB3D(2,:)=SF_AC_URB3D(1,:)
      LF_AC_URB3D(2,:)=LF_AC_URB3D(1,:)
      CM_AC_URB3D(2,:)=CM_AC_URB3D(1,:)
      SFVENT_URB3D(2,:)=SFVENT_URB3D(1,:)
      LFVENT_URB3D(2,:)=LFVENT_URB3D(1,:)
      EP_PV_URB3D(2,:)=EP_PV_URB3D(1,:)
      QGR_URB3D(2,:)=QGR_URB3D(1,:)
      TGR_URB3D(2,:)=TGR_URB3D(1,:)
      DRAINGR_URB3D(2,:)=DRAINGR_URB3D(1,:)
      LP_URB2D(2,:)=LP_URB2D(1,:)
      LB_URB2D(2,:)=LB_URB2D(1,:)
      HGT_URB2D(2,:)=HGT_URB2D(1,:)
      MH_URB2D(2,:)=MH_URB2D(1,:)
      STDH_URB2D(2,:)=STDH_URB2D(1,:)
      LF_URB2D_S(2,:)=LF_URB2D_S(1,:)
      Z0_URB2D(2,:)=Z0_URB2D(1,:)
      VEGFRA(2,:)=VEGFRA(1,:)
      UST(2,:)=UST(1,:)
      XLONG(2,:)=XLONG(1,:)
      XLAT(2,:)=XLAT(1,:)
      T3D(2,:,:)=T3D(1,:,:)
      QV3D(2,:,:)=QV3D(1,:,:)
      U_PHY(2,:,:)=U_PHY(1,:,:)
      V_PHY(2,:,:)=V_PHY(1,:,:)
      P8W3D(2,:,:)=P8W3D(1,:,:)
      DZ8W(2,:,:)=DZ8W(1,:,:)
      TH_PHY(2,:,:)=TH_PHY(1,:,:)
      RHO(2,:,:)=RHO(1,:,:)
      P_PHY(2,:,:)=P_PHY(1,:,:)
      A_U_BEP(2,:,:)=A_U_BEP(1,:,:)
      A_V_BEP(2,:,:)=A_V_BEP(1,:,:)
      A_T_BEP(2,:,:)=A_T_BEP(1,:,:)
      A_Q_BEP(2,:,:)=A_Q_BEP(1,:,:)
      A_E_BEP(2,:,:)=A_E_BEP(1,:,:)
      B_U_BEP(2,:,:)=B_U_BEP(1,:,:)
      B_V_BEP(2,:,:)=B_V_BEP(1,:,:)
      B_T_BEP(2,:,:)=B_T_BEP(1,:,:)
      B_Q_BEP(2,:,:)=B_Q_BEP(1,:,:)
      B_E_BEP(2,:,:)=B_E_BEP(1,:,:)
      DLG_BEP(2,:,:)=DLG_BEP(1,:,:)
      DL_U_BEP(2,:,:)=DL_U_BEP(1,:,:)
      SF_BEP(2,:,:)=SF_BEP(1,:,:)
      VL_BEP(2,:,:)=VL_BEP(1,:,:)
      TRL_URB3D(2,:,:)=TRL_URB3D(1,:,:)
      TBL_URB3D(2,:,:)=TBL_URB3D(1,:,:)
      TGL_URB3D(2,:,:)=TGL_URB3D(1,:,:)
      TGRL_URB3D(2,:,:)=TGRL_URB3D(1,:,:)
      SMR_URB3D(2,:,:)=SMR_URB3D(1,:,:)
      TSLB(2,:,:)=TSLB(1,:,:)
      SMOIS_URB(2,:,:)=SMOIS_URB(1,:,:)
      TRB_URB4D(2,:,:)=TRB_URB4D(1,:,:)
      TW1_URB4D(2,:,:)=TW1_URB4D(1,:,:)
      TW2_URB4D(2,:,:)=TW2_URB4D(1,:,:)
      TGB_URB4D(2,:,:)=TGB_URB4D(1,:,:)
      TLEV_URB3D(2,:,:)=TLEV_URB3D(1,:,:)
      QLEV_URB3D(2,:,:)=QLEV_URB3D(1,:,:)
      TW1LEV_URB3D(2,:,:)=TW1LEV_URB3D(1,:,:)
      TW2LEV_URB3D(2,:,:)=TW2LEV_URB3D(1,:,:)
      TGLEV_URB3D(2,:,:)=TGLEV_URB3D(1,:,:)
      TFLEV_URB3D(2,:,:)=TFLEV_URB3D(1,:,:)
      SFWIN1_URB3D(2,:,:)=SFWIN1_URB3D(1,:,:)
      SFWIN2_URB3D(2,:,:)=SFWIN2_URB3D(1,:,:)
      SFW1_URB3D(2,:,:)=SFW1_URB3D(1,:,:)
      SFW2_URB3D(2,:,:)=SFW2_URB3D(1,:,:)
      SFR_URB3D(2,:,:)=SFR_URB3D(1,:,:)
      SFG_URB3D(2,:,:)=SFG_URB3D(1,:,:)
      T_PV_URB3D(2,:,:)=T_PV_URB3D(1,:,:)
      TRV_URB4D(2,:,:)=TRV_URB4D(1,:,:)
      QR_URB4D(2,:,:)=QR_URB4D(1,:,:)
      DRAIN_URB4D(2,:,:)=DRAIN_URB4D(1,:,:)
      SFRV_URB3D(2,:,:)=SFRV_URB3D(1,:,:)
      LFRV_URB3D(2,:,:)=LFRV_URB3D(1,:,:)
      DGR_URB3D(2,:,:)=DGR_URB3D(1,:,:)
      DG_URB3D(2,:,:)=DG_URB3D(1,:,:)
      LFR_URB3D(2,:,:)=LFR_URB3D(1,:,:)
      LFG_URB3D(2,:,:)=LFG_URB3D(1,:,:)
      HI_URB2D(2,:,:)=HI_URB2D(1,:,:)
      LF_URB2D(2,:,:)=LF_URB2D(1,:,:)
    CALL urban_var_init(13, TSK, TSLB, TMN, IVGTYP,                     &
         ims, ime, jms, jme, kms, kme, NSOIL,                            &
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,                     &
         .FALSE., 1,                                                     &
         XXXR_URB2D, XXXB_URB2D, XXXG_URB2D, XXXC_URB2D,                 &
         TR_URB2D, TB_URB2D, TG_URB2D, TC_URB2D, QC_URB2D,               &
         TRL_URB3D, TBL_URB3D, TGL_URB3D,                                &
         SH_URB2D, LH_URB2D, G_URB2D, RN_URB2D, TS_URB2D,                &
         NMAP, NMAP, NMAP, NMAP, NMAP, NMAP, NMAP, NMAP, NMAP, NMAP, NMAP, NUHI, &
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
         FRC_URB2D, UTYPE_URB2D, 1)
    CALL dumpin(r, 'frc_urb', FRC_URB2D(1,1)); CALL dumpin(r, 'utype', REAL(UTYPE_URB2D(1,1)))
    CALL dumpstate(r, 0)

    DO s = 1, nsteps
      TSK = base_tsk; HFX = base_hfx; QFX = base_qfx; LH = base_lh; GRDFLX = base_grdflx
      ALBEDO = base_albedo; EMISS = base_emiss; QSFC = base_qsfc; ZNT = base_znt
      CHS = base_chs; CHS2 = base_chs2; CQS2 = base_cqs2; UST = base_ust
      IVGTYP(2,:)=IVGTYP(1,:)
      UTYPE_URB2D(2,:)=UTYPE_URB2D(1,:)
      COSZ_URB2D(2,:)=COSZ_URB2D(1,:)
      XLAT_URB2D(2,:)=XLAT_URB2D(1,:)
      SWDOWN(2,:)=SWDOWN(1,:)
      SWDDIR(2,:)=SWDDIR(1,:)
      SWDDIF(2,:)=SWDDIF(1,:)
      GLW(2,:)=GLW(1,:)
      RAINBL(2,:)=RAINBL(1,:)
      ZNT(2,:)=ZNT(1,:)
      TSK(2,:)=TSK(1,:)
      HFX(2,:)=HFX(1,:)
      QFX(2,:)=QFX(1,:)
      LH(2,:)=LH(1,:)
      GRDFLX(2,:)=GRDFLX(1,:)
      ALBEDO(2,:)=ALBEDO(1,:)
      EMISS(2,:)=EMISS(1,:)
      QSFC(2,:)=QSFC(1,:)
      TMN(2,:)=TMN(1,:)
      CMR_SFCDIF(2,:)=CMR_SFCDIF(1,:)
      CHR_SFCDIF(2,:)=CHR_SFCDIF(1,:)
      CMC_SFCDIF(2,:)=CMC_SFCDIF(1,:)
      CHC_SFCDIF(2,:)=CHC_SFCDIF(1,:)
      CMGR_SFCDIF(2,:)=CMGR_SFCDIF(1,:)
      CHGR_SFCDIF(2,:)=CHGR_SFCDIF(1,:)
      TR_URB2D(2,:)=TR_URB2D(1,:)
      TB_URB2D(2,:)=TB_URB2D(1,:)
      TG_URB2D(2,:)=TG_URB2D(1,:)
      TC_URB2D(2,:)=TC_URB2D(1,:)
      QC_URB2D(2,:)=QC_URB2D(1,:)
      UC_URB2D(2,:)=UC_URB2D(1,:)
      XXXR_URB2D(2,:)=XXXR_URB2D(1,:)
      XXXB_URB2D(2,:)=XXXB_URB2D(1,:)
      XXXG_URB2D(2,:)=XXXG_URB2D(1,:)
      XXXC_URB2D(2,:)=XXXC_URB2D(1,:)
      SH_URB2D(2,:)=SH_URB2D(1,:)
      LH_URB2D(2,:)=LH_URB2D(1,:)
      G_URB2D(2,:)=G_URB2D(1,:)
      RN_URB2D(2,:)=RN_URB2D(1,:)
      TS_URB2D(2,:)=TS_URB2D(1,:)
      PSIM_URB2D(2,:)=PSIM_URB2D(1,:)
      PSIH_URB2D(2,:)=PSIH_URB2D(1,:)
      U10_URB2D(2,:)=U10_URB2D(1,:)
      V10_URB2D(2,:)=V10_URB2D(1,:)
      GZ1OZ0_URB2D(2,:)=GZ1OZ0_URB2D(1,:)
      AKMS_URB2D(2,:)=AKMS_URB2D(1,:)
      TH2_URB2D(2,:)=TH2_URB2D(1,:)
      Q2_URB2D(2,:)=Q2_URB2D(1,:)
      UST_URB2D(2,:)=UST_URB2D(1,:)
      OMG_URB2D(2,:)=OMG_URB2D(1,:)
      CMCR_URB2D(2,:)=CMCR_URB2D(1,:)
      TGR_URB2D(2,:)=TGR_URB2D(1,:)
      DRELR_URB2D(2,:)=DRELR_URB2D(1,:)
      DRELB_URB2D(2,:)=DRELB_URB2D(1,:)
      DRELG_URB2D(2,:)=DRELG_URB2D(1,:)
      FLXHUMR_URB2D(2,:)=FLXHUMR_URB2D(1,:)
      FLXHUMB_URB2D(2,:)=FLXHUMB_URB2D(1,:)
      FLXHUMG_URB2D(2,:)=FLXHUMG_URB2D(1,:)
      FRC_URB2D(2,:)=FRC_URB2D(1,:)
      CHS(2,:)=CHS(1,:)
      CHS2(2,:)=CHS2(1,:)
      CQS2(2,:)=CQS2(1,:)
      SF_AC_URB3D(2,:)=SF_AC_URB3D(1,:)
      LF_AC_URB3D(2,:)=LF_AC_URB3D(1,:)
      CM_AC_URB3D(2,:)=CM_AC_URB3D(1,:)
      SFVENT_URB3D(2,:)=SFVENT_URB3D(1,:)
      LFVENT_URB3D(2,:)=LFVENT_URB3D(1,:)
      EP_PV_URB3D(2,:)=EP_PV_URB3D(1,:)
      QGR_URB3D(2,:)=QGR_URB3D(1,:)
      TGR_URB3D(2,:)=TGR_URB3D(1,:)
      DRAINGR_URB3D(2,:)=DRAINGR_URB3D(1,:)
      LP_URB2D(2,:)=LP_URB2D(1,:)
      LB_URB2D(2,:)=LB_URB2D(1,:)
      HGT_URB2D(2,:)=HGT_URB2D(1,:)
      MH_URB2D(2,:)=MH_URB2D(1,:)
      STDH_URB2D(2,:)=STDH_URB2D(1,:)
      LF_URB2D_S(2,:)=LF_URB2D_S(1,:)
      Z0_URB2D(2,:)=Z0_URB2D(1,:)
      VEGFRA(2,:)=VEGFRA(1,:)
      UST(2,:)=UST(1,:)
      XLONG(2,:)=XLONG(1,:)
      XLAT(2,:)=XLAT(1,:)
      T3D(2,:,:)=T3D(1,:,:)
      QV3D(2,:,:)=QV3D(1,:,:)
      U_PHY(2,:,:)=U_PHY(1,:,:)
      V_PHY(2,:,:)=V_PHY(1,:,:)
      P8W3D(2,:,:)=P8W3D(1,:,:)
      DZ8W(2,:,:)=DZ8W(1,:,:)
      TH_PHY(2,:,:)=TH_PHY(1,:,:)
      RHO(2,:,:)=RHO(1,:,:)
      P_PHY(2,:,:)=P_PHY(1,:,:)
      A_U_BEP(2,:,:)=A_U_BEP(1,:,:)
      A_V_BEP(2,:,:)=A_V_BEP(1,:,:)
      A_T_BEP(2,:,:)=A_T_BEP(1,:,:)
      A_Q_BEP(2,:,:)=A_Q_BEP(1,:,:)
      A_E_BEP(2,:,:)=A_E_BEP(1,:,:)
      B_U_BEP(2,:,:)=B_U_BEP(1,:,:)
      B_V_BEP(2,:,:)=B_V_BEP(1,:,:)
      B_T_BEP(2,:,:)=B_T_BEP(1,:,:)
      B_Q_BEP(2,:,:)=B_Q_BEP(1,:,:)
      B_E_BEP(2,:,:)=B_E_BEP(1,:,:)
      DLG_BEP(2,:,:)=DLG_BEP(1,:,:)
      DL_U_BEP(2,:,:)=DL_U_BEP(1,:,:)
      SF_BEP(2,:,:)=SF_BEP(1,:,:)
      VL_BEP(2,:,:)=VL_BEP(1,:,:)
      TRL_URB3D(2,:,:)=TRL_URB3D(1,:,:)
      TBL_URB3D(2,:,:)=TBL_URB3D(1,:,:)
      TGL_URB3D(2,:,:)=TGL_URB3D(1,:,:)
      TGRL_URB3D(2,:,:)=TGRL_URB3D(1,:,:)
      SMR_URB3D(2,:,:)=SMR_URB3D(1,:,:)
      TSLB(2,:,:)=TSLB(1,:,:)
      SMOIS_URB(2,:,:)=SMOIS_URB(1,:,:)
      TRB_URB4D(2,:,:)=TRB_URB4D(1,:,:)
      TW1_URB4D(2,:,:)=TW1_URB4D(1,:,:)
      TW2_URB4D(2,:,:)=TW2_URB4D(1,:,:)
      TGB_URB4D(2,:,:)=TGB_URB4D(1,:,:)
      TLEV_URB3D(2,:,:)=TLEV_URB3D(1,:,:)
      QLEV_URB3D(2,:,:)=QLEV_URB3D(1,:,:)
      TW1LEV_URB3D(2,:,:)=TW1LEV_URB3D(1,:,:)
      TW2LEV_URB3D(2,:,:)=TW2LEV_URB3D(1,:,:)
      TGLEV_URB3D(2,:,:)=TGLEV_URB3D(1,:,:)
      TFLEV_URB3D(2,:,:)=TFLEV_URB3D(1,:,:)
      SFWIN1_URB3D(2,:,:)=SFWIN1_URB3D(1,:,:)
      SFWIN2_URB3D(2,:,:)=SFWIN2_URB3D(1,:,:)
      SFW1_URB3D(2,:,:)=SFW1_URB3D(1,:,:)
      SFW2_URB3D(2,:,:)=SFW2_URB3D(1,:,:)
      SFR_URB3D(2,:,:)=SFR_URB3D(1,:,:)
      SFG_URB3D(2,:,:)=SFG_URB3D(1,:,:)
      T_PV_URB3D(2,:,:)=T_PV_URB3D(1,:,:)
      TRV_URB4D(2,:,:)=TRV_URB4D(1,:,:)
      QR_URB4D(2,:,:)=QR_URB4D(1,:,:)
      DRAIN_URB4D(2,:,:)=DRAIN_URB4D(1,:,:)
      SFRV_URB3D(2,:,:)=SFRV_URB3D(1,:,:)
      LFRV_URB3D(2,:,:)=LFRV_URB3D(1,:,:)
      DGR_URB3D(2,:,:)=DGR_URB3D(1,:,:)
      DG_URB3D(2,:,:)=DG_URB3D(1,:,:)
      LFR_URB3D(2,:,:)=LFR_URB3D(1,:,:)
      LFG_URB3D(2,:,:)=LFG_URB3D(1,:,:)
      HI_URB2D(2,:,:)=HI_URB2D(1,:,:)
      LF_URB2D(2,:,:)=LF_URB2D(1,:,:)
      CALL noahmp_urban(1, NSOIL, IVGTYP, s,                                        &
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
           NMAP, NMAP, NMAP, NMAP,                                                  &
           NMAP, NMAP, NMAP, NMAP,                                                  &
           NMAP, NMAP, NMAP,                                                        &
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
      ! module_surface_driver.F:3466-3500 for an ISURBAN cell (rural FVEG blend,
      ! then the SF_URBAN_PHYSICS==1 urban overwrite), PSFC = surface pressure.
      Q2  = (fveg*q2mv + (1.-fveg)*q2mb)*(1.-FRC_URB2D(1,1)) +   &
             Q2_URB2D(1,1)*FRC_URB2D(1,1)
      T2  = (fveg*t2mv + (1.-fveg)*t2mb)*(1.-FRC_URB2D(1,1)) +   &
             (TH2_URB2D(1,1)/((1.E5/psfc)**RCP))*FRC_URB2D(1,1)
      TH2 = T2*(1.E5/psfc)**RCP
      U10 = U10_URB2D(1,1)
      V10 = V10_URB2D(1,1)
      CALL d(r,s,'TILE2_HFX',HFX(2,1))
      CALL d(r,s,'TILE2_TSK',TSK(2,1))
      CALL d(r,s,'TILE2_U10_URB2D',U10_URB2D(2,1))
      CALL d(r,s,'TILE2_V10_URB2D',V10_URB2D(2,1))
      CALL d(r,s,'TILE2_UST',UST(2,1))
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
    CALL d(rr, ss, 'TR', TR_URB2D(1,1)); CALL d(rr, ss, 'TB', TB_URB2D(1,1))
    CALL d(rr, ss, 'TG', TG_URB2D(1,1)); CALL d(rr, ss, 'TC', TC_URB2D(1,1))
    CALL d(rr, ss, 'QC', QC_URB2D(1,1)); CALL d(rr, ss, 'UC', UC_URB2D(1,1))
    CALL d(rr, ss, 'XXXR', XXXR_URB2D(1,1)); CALL d(rr, ss, 'XXXC', XXXC_URB2D(1,1))
    CALL d(rr, ss, 'CMR', CMR_SFCDIF(1,1)); CALL d(rr, ss, 'CHR', CHR_SFCDIF(1,1))
    CALL d(rr, ss, 'CMC', CMC_SFCDIF(1,1)); CALL d(rr, ss, 'CHC', CHC_SFCDIF(1,1))
    CALL d(rr, ss, 'DRELR', DRELR_URB2D(1,1)); CALL d(rr, ss, 'DRELB', DRELB_URB2D(1,1))
    CALL d(rr, ss, 'DRELG', DRELG_URB2D(1,1)); CALL d(rr, ss, 'FLXHUMR', FLXHUMR_URB2D(1,1))
    CALL d(rr, ss, 'FLXHUMB', FLXHUMB_URB2D(1,1)); CALL d(rr, ss, 'FLXHUMG', FLXHUMG_URB2D(1,1))
    DO kk = 1, NL
      CALL dk(rr, ss, 'TRL', kk, TRL_URB3D(1,kk,1))
      CALL dk(rr, ss, 'TBL', kk, TBL_URB3D(1,kk,1))
      CALL dk(rr, ss, 'TGL', kk, TGL_URB3D(1,kk,1))
    END DO
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

END PROGRAM urban_oracle
