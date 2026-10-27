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
      PROGRAM residual

      USE wg_probe
      USE module_model_constants, ONLY: cpv_m => cpv, cliq_m => cliq, cice_m => cice, ep_2, SVP2, SVPT0, SVP3
      USE module_bl_mynnedmf_common, ONLY: t0c, tice, one, cpv, cliq, cice
      IMPLICIT NONE
      ! module_bl_mynnedmf.F90:278-301 (module PARAMETERs, verbatim)
      real, parameter :: pr = 0.74, g1 = 0.235, b1 = 24.0, b2 = 15.0, c2 = 0.729, c3 = 0.340, c4 = 0.0, c5 = 0.2
      real, parameter :: a1 = b1*( 1.0-3.0*g1 )/6.0
      real, parameter :: c1 = g1 -1.0/( 3.0*a1*2.88449914061481660)
      real, parameter :: a2 = a1*( g1-c1 )/( g1*pr )
      real, parameter :: g2 = b2/b1*( 1.0-c3 ) +2.0*a1/b1*( 3.0-2.0*c2 )
      real, parameter :: rhcrit = 0.83                  ! :3431
      real, parameter :: am_unst = 10., ah_unst = 34.   ! :7714
      real, parameter :: VKC = 0.40                     ! module_sf_noahmplsm.F:206
      real, parameter :: max_liq_mass_fraction = 0.4    ! module_sf_noahmplsm.F:7135
      ! module_mp_thompson.F PRIVATE parameters (verbatim)
      REAL, PARAMETER:: PI = 3.1415926536, rho_w = 1000.0, rho_i = 890.0
      REAL, DIMENSION(9), PARAMETER:: rho_g = (/50., 100., 200., 300., 400., 500., 600., 700., 800./)
      REAL, PARAMETER:: am_r = PI*rho_w/6.0, am_i = PI*rho_i/6.0, am_s = 0.069
      INTEGER, PARAMETER:: idx_bg1 = 5
      REAL, PARAMETER:: am_g5 = PI*rho_g(idx_bg1)/6.0
      REAL, PARAMETER:: mu_r = 0.0, mu_g = 0.0, mu_i = 0.0, bm_r = 3.0, bm_g = 3.0, bm_i = 3.0, bm_s = 2.0
      REAL, PARAMETER:: bv_s = 0.55, bv_i = 1.0, Kap0 = 490.6
      REAL, PARAMETER:: lsub = 2.834E6, lvap0 = 2.5E6, lfus = lsub - lvap0, olfus = 1./lfus
      REAL:: x
      REAL:: dts_a = 4.5, dts_b = 18., dts_c = 13.5, dampcoef_a = 0.2   ! namelist REAL variables
      REAL:: dt_54 = 54., dt_18 = 18., dt_6 = 6., snow_ret_fac = 5.e-5  ! DT; MPTABLE SNOW_RET_FAC
      REAL:: am_g(9), cie(7), cig(7), cre(13), crg(13), cge(12), cgg(12,1), cse(18), csg(18), oig1, org2, ogg2
      INTEGER:: n
      am_g(idx_bg1) = am_g5
      cie(1) = mu_i + 1.; cie(2) = bm_i + mu_i + 1.
      do n = 1, 2
         cig(n) = WGAMMA(cie(n))
      enddo
      oig1 = 1./cig(1)
      cre(2) = mu_r + 1.; cre(3) = bm_r + mu_r + 1.
      do n = 2, 3
         crg(n) = WGAMMA(cre(n))
      enddo
      org2 = 1./crg(2)
      cge(2) = mu_g + 1.; cge(3) = bm_g + mu_g + 1.
      cgg(2,1) = WGAMMA(cge(2)); cgg(3,1) = WGAMMA(cge(3))
      ogg2 = 1./cgg(2,1)
      cse(4) = bm_s + bv_s + 1.                          ! :730
      csg(4) = WGAMMA(cse(4))                            ! :745

      x = (t0c-6.) - tice
      write(*,'(I0,1X,I0)') 0, transfer(x, 1)
      x = t0c-6.
      write(*,'(I0,1X,I0)') 1, transfer(x, 1)
      x = cpv-cliq
      write(*,'(I0,1X,I0)') 2, transfer(x, 1)
      x = cpv-cice
      write(*,'(I0,1X,I0)') 3, transfer(x, 1)
      x = t0c - tice
      write(*,'(I0,1X,I0)') 4, transfer(x, 1)
      x = one-rhcrit
      write(*,'(I0,1X,I0)') 5, transfer(x, 1)
      x = one -c2
      write(*,'(I0,1X,I0)') 6, transfer(x, 1)
      x = b1*( g1+g2 ) -3.0*a1*( one -c2 )
      write(*,'(I0,1X,I0)') 7, transfer(x, 1)
      x = b1*( g1-c1 )
      write(*,'(I0,1X,I0)') 8, transfer(x, 1)
      x = 2.0*a1*(3.0-2.0*c2)
      write(*,'(I0,1X,I0)') 9, transfer(x, 1)
      x = 3.0*a2
      write(*,'(I0,1X,I0)') 10, transfer(x, 1)
      x = one-c5
      write(*,'(I0,1X,I0)') 11, transfer(x, 1)
      x = -0.33333*am_unst
      write(*,'(I0,1X,I0)') 12, transfer(x, 1)
      x = -0.33333*ah_unst
      write(*,'(I0,1X,I0)') 13, transfer(x, 1)
      x = 1./273.15
      write(*,'(I0,1X,I0)') 14, transfer(x, 1)
      x = 1.-ep_2
      write(*,'(I0,1X,I0)') 15, transfer(x, 1)
      x = VKC*VKC
      write(*,'(I0,1X,I0)') 16, transfer(x, 1)
      x = ( 2.71828 - 1.0 )
      write(*,'(I0,1X,I0)') 17, transfer(x, 1)
      x = max_liq_mass_fraction/(1.0 - max_liq_mass_fraction)
      write(*,'(I0,1X,I0)') 18, transfer(x, 1)
      x = (0.412 / 0.468)
      write(*,'(I0,1X,I0)') 19, transfer(x, 1)
      x = 0.2/1000.
      write(*,'(I0,1X,I0)') 20, transfer(x, 1)
      x = SVP2*SVPT0
      write(*,'(I0,1X,I0)') 21, transfer(x, 1)
      x = (SVP2*SVPT0)-SVP2*SVP3
      write(*,'(I0,1X,I0)') 22, transfer(x, 1)
      x = 1./((110000.0-5000.)/REAL(220-1))
      write(*,'(I0,1X,I0)') 23, transfer(x, 1)
      x = 1.0 / 0.278
      write(*,'(I0,1X,I0)') 24, transfer(x, 1)
      x = 1./am_s
      write(*,'(I0,1X,I0)') 25, transfer(x, 1)
      x = am_i*cig(2)*oig1
      write(*,'(I0,1X,I0)') 26, transfer(x, 1)
      x = am_r*crg(3)*org2
      write(*,'(I0,1X,I0)') 27, transfer(x, 1)
      x = am_g(idx_bg1)*cgg(3,1)*ogg2
      write(*,'(I0,1X,I0)') 28, transfer(x, 1)
      x = Kap0*csg(4)
      write(*,'(I0,1X,I0)') 29, transfer(x, 1)
      x = 4218.*olfus
      write(*,'(I0,1X,I0)') 30, transfer(x, 1)
      x = dts_a*dampcoef_a
      write(*,'(I0,1X,I0)') 31, transfer(x, 1)
      x = dts_b*dampcoef_a
      write(*,'(I0,1X,I0)') 32, transfer(x, 1)
      x = dts_c*dampcoef_a
      write(*,'(I0,1X,I0)') 33, transfer(x, 1)
      x = -0.01*dt_54/3600.0
      write(*,'(I0,1X,I0)') 34, transfer(x, 1)
      x = -0.01*dt_18/3600.0
      write(*,'(I0,1X,I0)') 35, transfer(x, 1)
      x = snow_ret_fac*dt_54
      write(*,'(I0,1X,I0)') 36, transfer(x, 1)
      x = snow_ret_fac*dt_6
      write(*,'(I0,1X,I0)') 37, transfer(x, 1)
      x = snow_ret_fac*dt_18
      write(*,'(I0,1X,I0)') 38, transfer(x, 1)
      END PROGRAM residual
