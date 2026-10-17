! WRF Thompson mp8 init/runtime REAL constants consumed by the gpuwrf release path. GAMMLN/WGAMMA are copied
! verbatim (sed) from phys/module_mp_thompson.F; every other line reproduces the cited WRF declaration/statement.
      MODULE wg_probe
      CONTAINS
      REAL FUNCTION GAMMLN(XX)
!     --- RETURNS THE VALUE LN(GAMMA(XX)) FOR XX > 0.
      IMPLICIT NONE
      REAL, INTENT(IN):: XX
      DOUBLE PRECISION, PARAMETER:: STP = 2.5066282746310005D0
      DOUBLE PRECISION, DIMENSION(6), PARAMETER:: &
               COF = (/76.18009172947146D0, -86.50532032941677D0, &
                       24.01409824083091D0, -1.231739572450155D0, &
                      .1208650973866179D-2, -.5395239384953D-5/)
      DOUBLE PRECISION:: SER,TMP,X,Y
      INTEGER:: J

      X=XX
      Y=X
      TMP=X+5.5D0
      TMP=(X+0.5D0)*LOG(TMP)-TMP
      SER=1.000000000190015D0
      DO 11 J=1,6
        Y=Y+1.D0
        SER=SER+COF(J)/Y
11    CONTINUE
      GAMMLN=TMP+LOG(STP*SER/X)
      END FUNCTION GAMMLN
      REAL FUNCTION WGAMMA(y)

      IMPLICIT NONE
      REAL, INTENT(IN):: y

      WGAMMA = EXP(GAMMLN(y))

      END FUNCTION WGAMMA
      END MODULE wg_probe
      PROGRAM wrfconst
      USE wg_probe
      IMPLICIT NONE
      REAL, PARAMETER:: PI = 3.1415926536                                   ! :67
      REAL, PARAMETER:: rho_w = 1000.0, rho_i = 890.0                       ! :70/:72
      REAL, DIMENSION(9), PARAMETER:: rho_g = (/50., 100., 200., 300., 400., 500., 600., 700., 800./)   ! :78
      REAL, PARAMETER:: am_r = PI*rho_w/6.0, am_i = PI*rho_i/6.0            ! :128/:137
      REAL, DIMENSION(9), PARAMETER:: am_g = (/PI*rho_g(1)/6.0, PI*rho_g(2)/6.0, PI*rho_g(3)/6.0, &
                 PI*rho_g(4)/6.0, PI*rho_g(5)/6.0, PI*rho_g(6)/6.0, PI*rho_g(7)/6.0, PI*rho_g(8)/6.0, PI*rho_g(9)/6.0 /)
      INTEGER, PARAMETER:: idx_bg1 = 5                                       ! :79
      REAL, PARAMETER:: mu_r = 0.0, mu_g = 0.0, mu_i = 0.0
      REAL, PARAMETER:: bm_r = 3.0, bm_g = 3.0, bm_i = 3.0
      REAL, PARAMETER:: av_r = 4854.0, bv_r = 1.0, av_s = 40.0, bv_s = 0.55
      REAL, PARAMETER:: av_g_old = 442.0, bv_g_old = 0.89, bv_i = 1.0, bv_c = 2.0, Sc = 0.632
      REAL, PARAMETER:: C_cube = 0.5, C_sqrd = 0.15, xm0i = 1.E-12, Nt_c = 100.E6
      REAL, PARAMETER:: rho_not = 101325.0/(287.05*298.0)                  ! :192
      REAL, PARAMETER:: lsub = 2.834E6, lvap0 = 2.5E6, lfus = lsub - lvap0, olfus = 1./lfus   ! :216-219
      REAL:: Sc3, D0i, cge(12), cgg(12), cre(13), crg(13), cie(7), cig(7), ccg(5), cce(5), ogg1, ogg2, ogg3
      REAL:: ocg1, ocg2, obmr, obmi, obmg, x, mvd
      DOUBLE PRECISION:: lamr, lamr3
      REAL, PARAMETER:: D0r = 50.E-6, D0s = 300.E-6                         ! :225/:226
      INTEGER:: n, nu_c
      Sc3 = Sc**(1./3.)                                                      ! :663
      D0i = (xm0i/am_i)**(1./bm_i)                                           ! :666
      nu_c = 12                                                              ! MIN(15, NINT(1000.E6/Nt_c) + 2)
      cce(1) = nu_c + 1.; cce(2) = bm_r + nu_c + 1.; cce(3) = bm_r + nu_c + 4.
      cce(4) = nu_c + bv_c + 1.; cce(5) = bm_r + nu_c + bv_c + 1.
      do n = 1, 5
         ccg(n) = WGAMMA(cce(n))
      enddo
      ocg1 = 1./ccg(1); ocg2 = 1./ccg(2)
      cie(1) = mu_i + 1.; cie(2) = bm_i + mu_i + 1.; cie(3) = bm_i + mu_i + bv_i + 1.; cie(4) = mu_i + bv_i + 1.
      cie(5) = mu_i + 2.; cie(6) = bm_i*0.5 + mu_i + bv_i + 1.; cie(7) = bm_i*0.5 + mu_i + 1.
      do n = 1, 7
         cig(n) = WGAMMA(cie(n))
      enddo
      cre(1) = bm_r + 1.; cre(2) = mu_r + 1.; cre(3) = bm_r + mu_r + 1.; cre(4) = bm_r*2. + mu_r + 1.
      cre(5) = mu_r + bv_r + 1.; cre(6) = bm_r + mu_r + bv_r + 1.; cre(7) = bm_r*0.5 + mu_r + bv_r + 1.
      cre(8) = bm_r + mu_r + bv_r + 3.; cre(9) = mu_r + bv_r + 3.; cre(10) = mu_r + 2.
      cre(11) = 0.5*(bv_r + 5. + 2.*mu_r); cre(12) = bm_r*0.5 + mu_r + 1.; cre(13) = bm_r*2. + mu_r + bv_r + 1.
      do n = 1, 13
         crg(n) = WGAMMA(cre(n))
      enddo
      cge(1) = bm_g + 1.; cge(2) = mu_g + 1.; cge(3) = bm_g + mu_g + 1.; cge(4) = bm_g*2. + mu_g + 1.
      cge(10) = mu_g + 2.; cge(12) = bm_g*0.5 + mu_g + 1.
      cge(5) = bm_g*2. + mu_g + bv_g_old + 1.; cge(6) = bm_g + mu_g + bv_g_old + 1.
      cge(7) = bm_g*0.5 + mu_g + bv_g_old + 1.; cge(8) = mu_g + bv_g_old + 1.; cge(9) = mu_g + bv_g_old + 3.
      cge(11) = 0.5*(bv_g_old + 5. + 2.*mu_g)
      do n = 1, 12
         cgg(n) = WGAMMA(cge(n))
      enddo
      ogg1 = 1./cgg(1); ogg2 = 1./cgg(2); ogg3 = 1./cgg(3)
      obmr = 1./bm_r; obmi = 1./bm_i; obmg = 1./bm_g
      call p('AM_R', am_r); call p('AM_I', am_i); call p('D0I', D0i); call p('AM_G_MP8', am_g(idx_bg1))
      call p('RHO_NOT', rho_not); call p('LFUS', lfus); call p('OLFUS', olfus); call p('SC3', Sc3)
      call p('OBMR', obmr); call p('OBMI', obmi); call p('OBMG', obmg)
      call p('CRE3', cre(3)); call p('CRE6', cre(6)); call p('CRE7', cre(7)); call p('CRE8', cre(8)); call p('CRE12', cre(12))
      call p('CIE2', cie(2)); call p('CGE9_OLD', cge(9)); call p('CGE11_OLD', cge(11))
      call p('CRG2', crg(2)); call p('CRG3', crg(3)); call p('CRG6', crg(6)); call p('CRG7', crg(7)); call p('CRG8', crg(8))
      call p('CRG9', crg(9)); call p('CRG10', crg(10)); call p('CRG11', crg(11)); call p('CRG12', crg(12))
      call p('ORG1', 1./crg(1)); call p('ORG2', 1./crg(2)); call p('ORG3', 1./crg(3))
      call p('CIG1', cig(1)); call p('CIG2', cig(2)); call p('CIG3', cig(3)); call p('CIG5', cig(5)); call p('CIG6', cig(6))
      call p('CIG7', cig(7)); call p('OIG1', 1./cig(1)); call p('OIG2', 1./cig(2))
      call p('CCG1_NU12', ccg(1)); call p('CCG2_NU12', ccg(2)); call p('CCG3_NU12', ccg(3)); call p('CCG4_NU12', ccg(4))
      call p('CCG5_NU12', ccg(5)); call p('OCG1_NU12', ocg1); call p('OCG2_NU12', ocg2)
      call p('CGG6_OLD', cgg(6)); call p('CGG9_OLD', cgg(9)); call p('CGG11_OLD', cgg(11)); call p('CGG10', cgg(10))
      call p('OGG3', ogg3)
      x = PI*.25*av_r * crg(9);                      call p('T1_QR_QC', x)          ! :786
      x = PI*.25*av_r * crg(9);                      call p('T1_QR_QI', x)          ! :787
      x = PI*.25*am_r*av_r * crg(8);                 call p('T2_QR_QI', x)          ! :788
      x = PI*.25*av_s;                               call p('T1_QS_QC', x)          ! :794
      x = PI*.25*av_s;                               call p('T1_QS_QI', x)          ! :797
      x = 0.78 * crg(10);                            call p('T1_QR_EV', x)          ! :800
      x = 0.308*Sc3*SQRT(av_r) * crg(11);            call p('T2_QR_EV', x)          ! :801
      x = 0.86;                                      call p('T1_SUBL_QS', x)        ! :804 t1_qs_sd
      x = 0.28*Sc3*SQRT(av_s);                       call p('T2_SUBL_QS', x)        ! :805 t2_qs_sd
      x = PI*4.*C_sqrd*olfus * 0.86;                 call p('T1_MELT_QS', x)        ! :808 t1_qs_me
      x = PI*4.*C_sqrd*olfus * 0.28*Sc3*SQRT(av_s);  call p('T2_MELT_QS', x)        ! :809 t2_qs_me
      x = 0.86 * cgg(10);                            call p('T1_SUBL_QG', x)        ! :812 t1_qg_sd
      x = PI*4.*C_cube*olfus * 0.86 * cgg(10);       call p('T1_MELT_QG', x)        ! :816 t1_qg_me
      x = PI*.25*av_g_old * cgg(9);                  call p('T1_QG_QC_OLD', x)      ! :2432/:2463 (idx_bg1)
      x = 0.28*Sc3*SQRT(av_g_old) * cgg(11);         call p('T2_SUBL_QG_OLD', x)    ! :2697/:2827
      x = PI*4.*C_cube*olfus * 0.28*Sc3*SQRT(av_g_old) * cgg(11); call p('T2_MELT_QG_OLD', x)  ! :2808
      x = Nt_c*am_r*ccg(2)*ocg1;                     call p('LAMC_PREFIX', x)       ! :1833/:2173/:3659 nc*am_r*ccg2*ocg1
      x = (ccg(3)*ocg2)**obmr;                       call p('DC_G_PREFIX', x)       ! :2181
      x = D0r*0.75;                                  call p('D0R_X075', x)          ! :1894/:3082/:3246/:4050
      x = 0.1*D0s;                                   call p('D0S_X01', x)           ! :2670
      x = am_r*nu_c*10.*D0r*D0r*D0r;                 call p('PNR_WAU_DEN', x)       ! :2191
      mvd = D0r*0.75
      lamr = (3.0 + mu_r + 0.672) / mvd                                       ! :1896 (lamr DOUBLE, RHS REAL)
      lamr3 = lamr**bm_r                                                      ! :1897 DOUBLE**REAL
      write(*,'(A,1X,I0,1X,ES26.17)') 'NR_MAX_LAMR3', transfer(lamr3, 1_8), lamr3
      CONTAINS
      SUBROUTINE p(name, v)
      CHARACTER(*), INTENT(IN):: name
      REAL, INTENT(IN):: v
      write(*,'(A,1X,I0,1X,ES26.17)') name, transfer(v, 1), v
      END SUBROUTINE p
      END PROGRAM wrfconst
