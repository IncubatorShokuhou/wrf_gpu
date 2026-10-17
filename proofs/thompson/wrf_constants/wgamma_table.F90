! WRF Thompson thompson_init gamma constants (mp8 path) with WRF's own GAMMLN/WGAMMA copied verbatim
! from phys/module_mp_thompson.F (lines below are extracted by sed, not edited).
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
      PROGRAM wg
      USE wg_probe
      IMPLICIT NONE
      REAL, PARAMETER:: PI = 3.1415926536
      REAL, PARAMETER:: rho_w = 1000.0
      REAL, PARAMETER:: am_r = PI*rho_w/6.0
      REAL, PARAMETER:: mu_r = 0.0, mu_g = 0.0, mu_i = 0.0, mu_s = 0.6357
      REAL, PARAMETER:: bm_r = 3.0, bm_s = 2.0, bm_g = 3.0, bm_i = 3.0
      REAL, PARAMETER:: av_r = 4854.0, bv_r = 1.0, av_s = 40.0, bv_s = 0.55
      REAL, PARAMETER:: av_g_old = 442.0, bv_g_old = 0.89, bv_i = 1.0, bv_c = 2.0, Sc = 0.632
      REAL:: cce(5,15), ccg(5,15), cie(7), cig(7), cre(13), crg(13), cse(17), csg(17), cge(12), cgg(12), Sc3, x
      INTEGER:: n
      do n = 1, 15
         cce(1,n) = n + 1.
         cce(2,n) = bm_r + n + 1.
         cce(3,n) = bm_r + n + 4.
         cce(4,n) = n + bv_c + 1.
         cce(5,n) = bm_r + n + bv_c + 1.
         ccg(1,n) = WGAMMA(cce(1,n)); ccg(2,n) = WGAMMA(cce(2,n)); ccg(3,n) = WGAMMA(cce(3,n))
         ccg(4,n) = WGAMMA(cce(4,n)); ccg(5,n) = WGAMMA(cce(5,n))
      enddo
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
      cse(1) = bm_s + 1.; cse(2) = bm_s + 2.; cse(3) = bm_s*2.; cse(4) = bm_s + bv_s + 1.
      cse(5) = bm_s*2. + bv_s + 1.; cse(6) = bm_s*2. + 1.; cse(7) = bm_s + mu_s + 1.; cse(8) = bm_s + mu_s + 2.
      cse(9) = bm_s + mu_s + 3.; cse(10) = bm_s + mu_s + bv_s + 1.; cse(11) = bm_s*2. + mu_s + bv_s + 1.
      cse(12) = bm_s*2. + mu_s + 1.; cse(13) = bv_s + 2.; cse(14) = bm_s + bv_s; cse(15) = mu_s + 1.
      cse(16) = 1.0 + (1.0 + bv_s)/2.; cse(17) = bm_s + bv_s + 2.
      do n = 1, 17
         csg(n) = WGAMMA(cse(n))
      enddo
      cge(1) = bm_g + 1.; cge(2) = mu_g + 1.; cge(3) = bm_g + mu_g + 1.; cge(4) = bm_g*2. + mu_g + 1.
      cge(10) = mu_g + 2.; cge(12) = bm_g*0.5 + mu_g + 1.
      cge(5) = bm_g*2. + mu_g + bv_g_old + 1.; cge(6) = bm_g + mu_g + bv_g_old + 1.
      cge(7) = bm_g*0.5 + mu_g + bv_g_old + 1.; cge(8) = mu_g + bv_g_old + 1.; cge(9) = mu_g + bv_g_old + 3.
      cge(11) = 0.5*(bv_g_old + 5. + 2.*mu_g)
      do n = 1, 12
         cgg(n) = WGAMMA(cge(n))
      enddo
      Sc3 = Sc**(1./3.)
      do n = 1, 15
         write(*,'(A,I0,A,I0,A,ES24.16,A,ES24.16,I12)') 'ccg(', 1, ',', n, ') arg', cce(1,n), ' val', ccg(1,n), transfer(ccg(1,n),1)
         write(*,'(A,I0,A,I0,A,ES24.16,A,ES24.16,I12)') 'ccg(', 2, ',', n, ') arg', cce(2,n), ' val', ccg(2,n), transfer(ccg(2,n),1)
         write(*,'(A,I0,A,I0,A,ES24.16,A,ES24.16,I12)') 'ccg(', 3, ',', n, ') arg', cce(3,n), ' val', ccg(3,n), transfer(ccg(3,n),1)
         write(*,'(A,I0,A,I0,A,ES24.16,A,ES24.16,I12)') 'ccg(', 4, ',', n, ') arg', cce(4,n), ' val', ccg(4,n), transfer(ccg(4,n),1)
         write(*,'(A,I0,A,I0,A,ES24.16,A,ES24.16,I12)') 'ccg(', 5, ',', n, ') arg', cce(5,n), ' val', ccg(5,n), transfer(ccg(5,n),1)
      enddo
      do n = 1, 7
         write(*,'(A,I0,A,ES24.16,A,ES24.16,I12)') 'cig(', n, ') arg', cie(n), ' val', cig(n), transfer(cig(n),1)
      enddo
      do n = 1, 13
         write(*,'(A,I0,A,ES24.16,A,ES24.16,I12)') 'crg(', n, ') arg', cre(n), ' val', crg(n), transfer(crg(n),1)
      enddo
      do n = 1, 17
         write(*,'(A,I0,A,ES24.16,A,ES24.16,I12)') 'csg(', n, ') arg', cse(n), ' val', csg(n), transfer(csg(n),1)
      enddo
      do n = 1, 12
         write(*,'(A,I0,A,ES24.16,A,ES24.16,I12)') 'cgg(', n, ') arg', cge(n), ' val', cgg(n), transfer(cgg(n),1)
      enddo
      x = PI*.25*av_r * crg(9);        write(*,'(A,ES24.16,I12)') 't1_qr_qc ', x, transfer(x,1)
      x = PI*.25*am_r*av_r * crg(8);   write(*,'(A,ES24.16,I12)') 't2_qr_qi ', x, transfer(x,1)
      x = 0.78 * crg(10);              write(*,'(A,ES24.16,I12)') 't1_qr_ev ', x, transfer(x,1)
      x = 0.308*Sc3*SQRT(av_r) * crg(11); write(*,'(A,ES24.16,I12)') 't2_qr_ev ', x, transfer(x,1)
      x = 1./crg(1); write(*,'(A,ES24.16,I12)') 'org1 ', x, transfer(x,1)
      x = 1./crg(2); write(*,'(A,ES24.16,I12)') 'org2 ', x, transfer(x,1)
      x = 1./crg(3); write(*,'(A,ES24.16,I12)') 'org3 ', x, transfer(x,1)
      x = 1./cig(1); write(*,'(A,ES24.16,I12)') 'oig1 ', x, transfer(x,1)
      x = 1./cig(2); write(*,'(A,ES24.16,I12)') 'oig2 ', x, transfer(x,1)
      x = 1./cgg(1); write(*,'(A,ES24.16,I12)') 'ogg1 ', x, transfer(x,1)
      x = 1./cgg(2); write(*,'(A,ES24.16,I12)') 'ogg2 ', x, transfer(x,1)
      x = 1./cgg(3); write(*,'(A,ES24.16,I12)') 'ogg3 ', x, transfer(x,1)
      write(*,'(A,ES24.16,I12)') 'Sc3 ', Sc3, transfer(Sc3,1)
      END PROGRAM wg
