! Standalone SMS tke_km + tke_dissip, linked to unmodified pristine libwrflib.a.
! Derived from proofs/o1_smag3d's caller/halo driver. Deliberately stops BEFORE
! phy_bc/nonlocal/vertical_diffusion_implicit; hpbl/dlk supplied independent operands.
! Header km_opt=2 is only the shared input format; execute km_opt=5 after reading.
! No solve_em invocation, so no E104 shared hard-coded dump path is reachable.
PROGRAM sms_upstream_driver
  USE module_model_constants, ONLY : ep_1, ep_2, karman, g, cp
  USE module_configure, ONLY : grid_config_rec_type
  USE module_state_description, ONLY : p_qv, p_qc, p_qr, p_qi, p_qs, param_first_scalar
  USE module_bc, ONLY : set_physical_bc3d, set_physical_bc2d
  USE module_diffusion_em, ONLY : compute_diff_metrics, cal_deform_and_div, calculate_km_kh, &
                                  tke_dissip, nonlocal_flux, free_atmos_length, meso_length_scale, pthnl, pu
  IMPLICIT NONE
  TYPE(grid_config_rec_type) :: cf
  INTEGER, PARAMETER :: H = 5
  INTEGER :: hdr(16)
  REAL :: rhdr(16)
  INTEGER :: nx, ny, nz, n_moist, im
  INTEGER :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  REAL :: dx, dy, dt, rdx, rdy, cf1, cf2, cf3, khdif, kvdif, dampcoef, zdamp, mix_upper_bound
  REAL :: tmpdz, thetasfc, thetatop
  INTEGER :: i,j,k,is,ie,js,je
  REAL, ALLOCATABLE :: dth_save(:,:,:),elmin(:,:,:),uphy(:,:,:),vphy(:,:,:)
  REAL, ALLOCATABLE :: rmol(:,:),br(:,:),wspd(:,:),hfx_start(:,:),gu_start(:,:),gv_start(:,:)
  INTEGER, ALLOCATABLE :: kpbl(:,:)
  LOGICAL :: warm_rain
  CHARACTER(LEN=512) :: fin, fout
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: u, v, w, thp, th_phy, t_phy, p_phy, p8w, t8w, ph, phb, rho, tke, &
       z, rdz, rdzw, zx, zy, div, d11, d22, d33, d12, d13, d23, bn2, xkmh, xkmv, xkhh, xkhv, &
       ru_t, rv_t, rw_t, t_t, tke_t, dlk, xkmv_meso, rublten, rvblten, rucuten, rvcuten, rushten, rvshten, &
       l_diss, nlflux
  REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: moist, moist_t, nba_rij, nba_mij, chem, chem_t, scal, scal_t, trac, trac_t
  REAL, ALLOCATABLE, DIMENSION(:,:) :: msftx, msfty, msfux, msfuy, msfvx, msfvy, ust, ustm, hfx, qfx, pblh, gamu, gamv, mut
  REAL, ALLOCATABLE, DIMENSION(:) :: fnm, fnp, dn, dnw, u_base, v_base, t_base, qv_base, c1h, c2h

  CALL get_command_argument(1, fin)
  CALL get_command_argument(2, fout)
  OPEN(10, FILE=TRIM(fin), ACCESS='stream', FORM='unformatted', STATUS='old')
  READ(10) hdr
  READ(10) rhdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); n_moist = hdr(4)
  dx = rhdr(1); dy = rhdr(2); dt = rhdr(3); rdx = rhdr(4); rdy = rhdr(5)
  cf1 = rhdr(6); cf2 = rhdr(7); cf3 = rhdr(8); mix_upper_bound = rhdr(10)

  ! ---- config_flags: every component read by module_diffusion_em / module_bc ----
  cf%specified = (hdr(5) == 1); cf%nested = (hdr(5) == 2)
  cf%periodic_x = (hdr(6) == 1); cf%periodic_y = (hdr(7) == 1)
  cf%open_xs = .FALSE.; cf%open_xe = .FALSE.; cf%open_ys = .FALSE.; cf%open_ye = .FALSE.
  cf%symmetric_xs = .FALSE.; cf%symmetric_xe = .FALSE.; cf%symmetric_ys = .FALSE.; cf%symmetric_ye = .FALSE.
  cf%polar = .FALSE.
  cf%mix_isotropic = hdr(8)
  cf%isfflx = hdr(9)
  cf%bl_pbl_physics = hdr(10)
  cf%use_theta_m = hdr(11)
  cf%mix_full_fields = (hdr(12) == 1)
  cf%km_opt = hdr(13)
  cf%diff_opt = 2
  cf%sfs_opt = 0; cf%m_opt = 0
  cf%cu_physics = 0; cf%shcu_physics = 0
  cf%moist_mix2_off = (hdr(14) == 1); cf%tke_mix2_off = (hdr(15) == 1)
  cf%chem_mix2_off = .FALSE.; cf%scalar_mix2_off = .FALSE.; cf%tracer_mix2_off = .FALSE.
  cf%spec_bdy_width = 5
  cf%c_s = rhdr(9); cf%c_k = rhdr(13)
  cf%tke_drag_coefficient = rhdr(11); cf%tke_heat_flux = rhdr(12)
  cf%damp_opt = 0
  khdif = 0.; kvdif = 0.; dampcoef = 0.; zdamp = 5000.
  warm_rain = .FALSE.
  ! Thompson-like moist slots (calculate_N2 sums P_QV/P_QC/P_QI).
  p_qv = 2; p_qc = 3; p_qr = 4; p_qi = 5; p_qs = 6

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = ids - H; ime = ide + H; jms = jds - H; jme = jde + H; kms = kds; kme = kde
  its = ids; ite = ide; jts = jds; jte = jde; kts = kds; kte = kde

  CALL a3(u); CALL a3(v); CALL a3(w); CALL a3(thp); CALL a3(th_phy); CALL a3(t_phy); CALL a3(p_phy)
  CALL a3(p8w); CALL a3(t8w); CALL a3(ph); CALL a3(phb); CALL a3(rho); CALL a3(tke)
  CALL a3(z); CALL a3(rdz); CALL a3(rdzw); CALL a3(zx); CALL a3(zy); CALL a3(div)
  CALL a3(d11); CALL a3(d22); CALL a3(d33); CALL a3(d12); CALL a3(d13); CALL a3(d23)
  CALL a3(bn2); CALL a3(xkmh); CALL a3(xkmv); CALL a3(xkhh); CALL a3(xkhv)
  CALL a3(ru_t); CALL a3(rv_t); CALL a3(rw_t); CALL a3(t_t); CALL a3(tke_t); CALL a3(dlk); CALL a3(xkmv_meso)
  CALL a3(rublten); CALL a3(rvblten); CALL a3(rucuten); CALL a3(rvcuten); CALL a3(rushten); CALL a3(rvshten)
  CALL a3(l_diss); CALL a3(nlflux); CALL a3(dth_save)
  ALLOCATE(moist(ims:ime,kms:kme,jms:jme,n_moist), moist_t(ims:ime,kms:kme,jms:jme,n_moist)); moist = 0.; moist_t = 0.
  ALLOCATE(nba_rij(ims:ime,kms:kme,jms:jme,1), nba_mij(ims:ime,kms:kme,jms:jme,1)); nba_rij = 0.; nba_mij = 0.
  ALLOCATE(chem(ims:ime,kms:kme,jms:jme,1), chem_t(ims:ime,kms:kme,jms:jme,1)); chem = 0.; chem_t = 0.
  ALLOCATE(scal(ims:ime,kms:kme,jms:jme,1), scal_t(ims:ime,kms:kme,jms:jme,1)); scal = 0.; scal_t = 0.
  ALLOCATE(trac(ims:ime,kms:kme,jms:jme,1), trac_t(ims:ime,kms:kme,jms:jme,1)); trac = 0.; trac_t = 0.
  CALL a2(msftx); CALL a2(msfty); CALL a2(msfux); CALL a2(msfuy); CALL a2(msfvx); CALL a2(msfvy)
  CALL a2(ust); CALL a2(ustm); CALL a2(hfx); CALL a2(qfx); CALL a2(pblh); CALL a2(gamu); CALL a2(gamv); CALL a2(mut)
  ALLOCATE(fnm(kms:kme), fnp(kms:kme), dn(kms:kme), dnw(kms:kme), u_base(kms:kme), v_base(kms:kme), &
           t_base(kms:kme), qv_base(kms:kme), c1h(kms:kme), c2h(kms:kme))
  c1h = 0.; c2h = 0.

  ! ---- inputs (Fortran order, interior extents) ----
  READ(10) u(ids:ide, 1:nz, jds:jde-1)
  READ(10) v(ids:ide-1, 1:nz, jds:jde)
  READ(10) w(ids:ide-1, 1:nz+1, jds:jde-1)
  READ(10) thp(ids:ide-1, 1:nz, jds:jde-1)
  READ(10) th_phy(ids:ide-1, 1:nz, jds:jde-1)
  READ(10) t_phy(ids:ide-1, 1:nz, jds:jde-1)
  READ(10) p_phy(ids:ide-1, 1:nz, jds:jde-1)
  READ(10) p8w(ids:ide-1, 1:nz+1, jds:jde-1)
  READ(10) t8w(ids:ide-1, 1:nz+1, jds:jde-1)
  READ(10) ph(ids:ide-1, 1:nz+1, jds:jde-1)
  READ(10) phb(ids:ide-1, 1:nz+1, jds:jde-1)
  READ(10) rho(ids:ide-1, 1:nz, jds:jde-1)
  DO im = param_first_scalar, n_moist
    READ(10) moist(ids:ide-1, 1:nz, jds:jde-1, im)
  END DO
  READ(10) msftx(ids:ide-1, jds:jde-1); READ(10) msfty(ids:ide-1, jds:jde-1)
  READ(10) msfux(ids:ide, jds:jde-1);   READ(10) msfuy(ids:ide, jds:jde-1)
  READ(10) msfvx(ids:ide-1, jds:jde);   READ(10) msfvy(ids:ide-1, jds:jde)
  READ(10) ust(ids:ide-1, jds:jde-1); READ(10) hfx(ids:ide-1, jds:jde-1); READ(10) qfx(ids:ide-1, jds:jde-1)
  READ(10) fnm; READ(10) fnp; READ(10) dn; READ(10) dnw
  READ(10) u_base; READ(10) v_base; READ(10) t_base; READ(10) qv_base
  IF (cf%km_opt == 2) THEN
    READ(10) tke(ids:ide-1, 1:nz, jds:jde-1)
    READ(10) mut(ids:ide-1, jds:jde-1)
    READ(10) c1h; READ(10) c2h
  END IF
  READ(10) pblh(ids:ide-1,jds:jde-1)
  READ(10) dlk(ids:ide-1,1:nz,jds:jde-1)
  CALL a2(rmol);CALL a2(br);CALL a2(wspd);CALL a2(hfx_start);CALL a2(gu_start);CALL a2(gv_start)
  CALL a3(elmin);CALL a3(uphy);CALL a3(vphy)
  ALLOCATE(kpbl(ims:ime,jms:jme));kpbl=1
  READ(10) rmol(ids:ide-1,jds:jde-1),br(ids:ide-1,jds:jde-1),wspd(ids:ide-1,jds:jde-1)
  READ(10) gamu(ids:ide-1,jds:jde-1),gamv(ids:ide-1,jds:jde-1)
  hfx_start=hfx;gu_start=gamu;gv_start=gamv
  CLOSE(10)
  cf%km_opt = 5
  ustm = ust

  ! ---- WRF serial halo/BC fill of the inputs ----
  CALL bc3(u, 'u'); CALL bc3(v, 'v'); CALL bc3(w, 'w'); CALL bc3(ph, 'w'); CALL bc3(phb, 'w')
  CALL bc3(thp, 't'); CALL bc3(th_phy, 't'); CALL bc3(t_phy, 't'); CALL bc3(p_phy, 't')
  CALL bc3(p8w, 'w'); CALL bc3(t8w, 'w'); CALL bc3(rho, 't'); CALL bc3(tke, 't'); CALL bc2(mut, 't')
  CALL bc2(msftx, 't'); CALL bc2(msfty, 't'); CALL bc2(msfux, 'u'); CALL bc2(msfuy, 'u')
  CALL bc2(msfvx, 'v'); CALL bc2(msfvy, 'v')
  DO im = param_first_scalar, n_moist
    CALL set_physical_bc3d(moist(:,:,:,im), 't', cf, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                           ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
  END DO

  ! ---- first_rk_step_part2 diff_opt=2 sequence ----
  CALL compute_diff_metrics(cf, ph, phb, z, rdz, rdzw, zx, zy, rdx, rdy, &
                            ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  CALL bc3(rdzw, 'w'); CALL bc3(rdz, 'w'); CALL bc3(z, 'w'); CALL bc3(zx, 'e'); CALL bc3(zy, 'f')
  CALL set_physical_bc2d(ustm, 't', cf, ids, ide, jds, jde, ims, ime, jms, jme, ids, ide, jds, jde, its, ite, jts, jte)
  CALL set_physical_bc2d(ust, 't', cf, ids, ide, jds, jde, ims, ime, jms, jme, ids, ide, jds, jde, its, ite, jts, jte)
  CALL cal_deform_and_div(cf, u, v, w, div, d11, d22, d33, d12, d13, d23, nba_rij, 1, u_base, v_base, &
                          msfux, msfuy, msfvx, msfvy, msftx, msfty, rdx, rdy, dn, dnw, rdz, rdzw, fnm, fnp, &
                          cf1, cf2, cf3, zx, zy, &
                          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  DO j=jds,jde-1
  DO k=kts,nz
  DO i=ids,ide-1
    uphy(i,k,j)=0.5*(u(i,k,j)+u(i+1,k,j))
    vphy(i,k,j)=0.5*(v(i,k,j)+v(i,k,j+1))
  ENDDO
  ENDDO
  ENDDO
  CALL nonlocal_flux_probe(cf,nlflux,gamu,gamv,pblh,kpbl,dx,dy,dt,ust,hfx,qfx,br,ep_1,ep_2,karman, &
                    uphy,vphy,th_phy,rho,moist,n_moist,msftx,msfty,rdzw, &
                    ust,ust,wspd, &
                    ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  CALL free_atmos_length(cf,dx,dy,rdzw,rdz,tke,th_phy,elmin,hfx,qfx,moist,n_moist, &
                    ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  CALL meso_length_scale(cf,dx,dy,rdzw,rdz,tke,p8w,t8w,th_phy,dlk,pblh,elmin,rmol,rho,hfx,qfx,moist,n_moist, &
                    ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  CALL calculate_km_kh(cf, dt, dampcoef, zdamp, 0, xkmh, xkmv, xkhh, xkhv, bn2, khdif, kvdif, div, &
                       d11, d22, d33, d12, d13, d23, tke, p8w, t8w, th_phy, t_phy, p_phy, moist, dn, dnw, &
                       dx, dy, rdz, rdzw, cf%mix_isotropic, n_moist, cf1, cf2, cf3, warm_rain, mix_upper_bound, &
                       msftx, msfty, zx, zy, pblh, dlk, xkmv_meso, &
                       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  ! Save common tke_km preprocessing separately (verbatim loop ranges/REAL expressions).
  is=its; ie=MIN(ite,ide-1); js=jts; je=MIN(jte,jde-1)
  IF (cf%specified) THEN
    is=MAX(ids+1,its);ie=MIN(ide-2,ite);js=MAX(jds+1,jts);je=MIN(jde-2,jte)
  ENDIF
  DO j=js,je
  DO k=kts+1,nz-1
  DO i=is,ie
    tmpdz = 1.0/rdz(i,k+1,j) + 1.0/rdz(i,k,j)
    dth_save(i,k,j) = (th_phy(i,k+1,j) - th_phy(i,k-1,j))/tmpdz
  ENDDO
  ENDDO
  ENDDO
  k=kts
  DO j=js,je
  DO i=is,ie
    tmpdz=1.0/rdzw(i,k+1,j)+1.0/rdzw(i,k,j)
    thetasfc=t8w(i,kts,j)/(p8w(i,k,j)/100000.)**(287./1004.5)
    dth_save(i,k,j)=(th_phy(i,k+1,j)-thetasfc)/tmpdz
  ENDDO
  ENDDO
  k=nz
  DO j=js,je
  DO i=is,ie
    tmpdz=1.0/rdz(i,k,j)+0.5/rdzw(i,k,j)
    thetatop=t8w(i,kde,j)/(p8w(i,kde,j)/100000.)**(287./1004.5)
    dth_save(i,k,j)=(thetatop-th_phy(i,k-1,j))/tmpdz
  ENDDO
  ENDDO
  ! Component boundary: coefficients BEFORE phy_bc; SMS dissipation only stores l_diss.
  CALL tke_dissip(tke_t, cf, mut, c1h, c2h, tke, bn2, th_phy, p8w, t8w, z, &
                   dx, dy, rdz, rdzw, cf%mix_isotropic, msftx, msfty, pblh, dlk, l_diss, &
                   ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  IF (ANY(tke_t /= 0.)) STOP 'SMS dissipation must not add an explicit TKE sink'
  OPEN(11, FILE=TRIM(fout), ACCESS='stream', FORM='unformatted', STATUS='replace')
  WRITE(11) ims, ime, jms, jme, kms, kme
  WRITE(11) nlflux,dlk,elmin,rdz,rdzw,th_phy,tke,moist(:,:,:,p_qv),rho,uphy,vphy,p8w,t8w
  WRITE(11) gamu,gamv,pblh,hfx,rmol,br,wspd,ust,msftx,msfty,hfx_start,gu_start,gv_start
  CLOSE(11)
  PRINT *, 'SMS COEFFICIENT ORACLE OK ', nx, ny, nz


CONTAINS
  SUBROUTINE a3(a)
    REAL, ALLOCATABLE, INTENT(INOUT) :: a(:,:,:)
    ALLOCATE(a(ims:ime, kms:kme, jms:jme)); a = 0.
  END SUBROUTINE a3
  SUBROUTINE a2(a)
    REAL, ALLOCATABLE, INTENT(INOUT) :: a(:,:)
    ALLOCATE(a(ims:ime, jms:jme)); a = 0.
  END SUBROUTINE a2
  SUBROUTINE bc3(a, stag)
    REAL, INTENT(INOUT) :: a(ims:ime, kms:kme, jms:jme)
    CHARACTER(LEN=1), INTENT(IN) :: stag
    CALL set_physical_bc3d(a, stag, cf, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                           ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
  END SUBROUTINE bc3
  SUBROUTINE bc2(a, stag)
    REAL, INTENT(INOUT) :: a(ims:ime, jms:jme)
    CHARACTER(LEN=1), INTENT(IN) :: stag
    CALL set_physical_bc2d(a, stag, cf, ids, ide, jds, jde, ims, ime, jms, jme, ids, ide, jds, jde, its, ite, jts, jte)
  END SUBROUTINE bc2
 SUBROUTINE nonlocal_flux_probe (config_flags,nlflux,gamu,gamv,hpbl,kpbl,     &
                                  dx,dy,dt,ust,hfx,qfx,br,ep1,ep2,      &
                                  karman,u_phy,v_phy,th_phy,rho,moist,n_moist, &
                                  msftx,msfty,rdzw,                            &
                                  u10,v10,wspd,                                &
                                  ids, ide, jds, jde, kds, kde,     &
                                  ims, ime, jms, jme, kms, kme,     &
                                  its, ite, jts, jte, kts, kte      )
!-----------------------------------------------------------------------
!This subroutine prescibes the nonlocal heat flux profile based on LES analysis
!And compute the nonlocal momentum gamma term 
!References:
! Shin and Hong 2015, MWR
! Xu Zhang et al. 2018, MWR  
!-----------------------------------------------------------------------
! Begin declarations.
 
   IMPLICIT NONE
 
   TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags
 
   INTEGER ,         INTENT(IN   ) ::       ids, ide, jds, jde, kds, kde, &
                                            ims, ime, jms, jme, kms, kme, &
                                            its, ite, jts, jte, kts, kte

   INTEGER ,         INTENT(IN   ) :: n_moist

   REAL,    INTENT(IN )     ::      ep1,ep2,karman

   REAL,    INTENT(IN )     ::      dx,dy,dt

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme ), INTENT(IN) ::   &
   th_phy,u_phy,v_phy

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme ), INTENT(IN) :: rdzw

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme, n_moist ), INTENT( INOUT ) :: &
   moist

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme ), INTENT(IN) :: rho
 
   REAL, DIMENSION( ims:ime, jms:jme ), INTENT(INOUT)   :: hfx,qfx,br,ust,hpbl

   REAL, DIMENSION( ims:ime, jms:jme ), INTENT(IN   )   :: u10,v10,wspd

   REAL, DIMENSION( ims:ime, jms:jme ), INTENT(IN   )   :: msftx,msfty

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme ), INTENT(OUT) :: nlflux

   REAL, DIMENSION( ims:ime, jms:jme ), INTENT(OUT   )  :: gamu,gamv

   INTEGER, DIMENSION( ims:ime, jms:jme ), INTENT(INOUT   )  :: kpbl

! Local variables.
  REAL, DIMENSION( its:ite, kts:kte,   jts:jte ) :: zq
  REAL, DIMENSION( its:ite, kts:kte-1, jts:jte ) :: za,thv
  REAL, DIMENSION( its:ite, kts:kte,   jts:jte ) :: zfacmf,entfacmf
  REAL, DIMENSION( its:ite, jts:jte )            :: govrth,sflux,wstar3,wstar,rigs
  LOGICAL, DIMENSION( its:ite,jts:jte )          :: pblflg, sfcflg, stable
  REAL, DIMENSION( its:ite, jts:jte )            :: deltaoh,we,enlfrac2
  REAL, DIMENSION( its:ite, jts:jte )            :: hfxpbl,bfxpbl
  REAL, DIMENSION( its:ite, jts:jte )            :: dthv,wm2
  REAL, DIMENSION( its:ite, jts:jte )            :: wscale,thermal
  REAL               :: tvcon,delb,cpm,wm3,bfx0,ust3
  REAL               :: dtheta,du,dv,thsfc,brint
  INTEGER            :: i,j,k,i_start,i_end,j_start,j_end,ktf
  REAL               :: mlfrac,ezfrac,sfcfracn,sflux0,snlflux0
  REAL               :: amf1,amf2,amf3,bmf2,bmf3,pth1,delxy,pu1
  REAL               :: heat_flux
  REAL,PARAMETER     :: phifac = 8.,sfcfrac = 0.1,d1 = 0.02, d2 = 0.05, zfmin = 1.e-8
  REAL,PARAMETER     :: h1 = 0.33333335, h2 = 0.6666667, tmin = 1.e-2
! tunable parameters
  REAL,PARAMETER     ::  mltop = 1.0, sfcfracn1 = 0.075
  REAL,PARAMETER     ::  nlfrac = 0.68 ! the ratio of nonlocal heat flux to total heat flux
  REAL,PARAMETER     ::  enlfrac = -0.15 
  REAL,PARAMETER     ::  a11 = 1.0, a12 = -1.15
  REAL,PARAMETER     ::  ezfac = 1.5
  REAL,PARAMETER     ::  cpent = -0.4, rigsmax = 100., rimin = -100.
  REAL,PARAMETER     ::  entfmin = 1.0, entfmax = 5.0, sm = 10.9

   OPEN(17,FILE='<USER_HOME>/wrf_gpu2_lanes/s1-km5/oracle/nonlocal_probe.bin',ACCESS='stream',FORM='unformatted',STATUS='replace')
   ktf=MIN(kte,kde-1)
 
   i_start = its
   i_end   = MIN(ite,ide-1)
   j_start = jts
   j_end   = MIN(jte,jde-1)
 
   IF ( config_flags%open_xs .or. config_flags%specified .or. &
        config_flags%nested) i_start = MAX(ids+1,its)
   IF ( config_flags%open_xe .or. config_flags%specified .or. &
        config_flags%nested) i_end   = MIN(ide-2,ite)
   IF ( config_flags%open_ys .or. config_flags%specified .or. &
        config_flags%nested) j_start = MAX(jds+1,jts)
   IF ( config_flags%open_ye .or. config_flags%specified .or. &
        config_flags%nested) j_end   = MIN(jde-2,jte)
   IF ( config_flags%periodic_x ) i_start = its
   IF ( config_flags%periodic_x ) i_end = MIN(ite,ide-1)
 
   DO j = j_start, j_end
   DO i = i_start, i_end
       zq(i,1,j) = 0.0
   ENDDO
   ENDDO

   DO j = j_start, j_end
   DO k = kts, ktf
   DO i = i_start, i_end
         zq(i,k+1,j) = 1.0/rdzw(i,k,j) + zq(i,k,j)
   ENDDO
   ENDDO
   ENDDO
 
   DO j = j_start,j_end
   DO k = kts,ktf
   DO i = i_start,i_end
     za(i,k,j) = 0.5*(zq(i,k,j) + zq(i,k+1,j))
   ENDDO
   ENDDO
   ENDDO
 
   DO j = j_start, j_end
   DO i = i_start, i_end
     deltaoh(i,j)  = 0.0
     rigs (i,j)    = 0.0
     enlfrac2(i,j) = 0.0
   ENDDO
   ENDDO

   DO j = j_start, j_end
   DO k = kts, ktf
   DO i = i_start, i_end
      zfacmf(i,k,j) = 0.0
   ENDDO
   ENDDO
   ENDDO
 
   DO j = j_start, j_end
   DO k = kts, ktf
   DO i = i_start, i_end
      nlflux(i,k,j) = 0.0
   ENDDO
   ENDDO
   ENDDO
 
   DO j = j_start, j_end
   DO k = kts, ktf
   DO i = i_start, i_end
      tvcon = 1. + ep1*moist(i,k,j,P_QV)
      thv(i,k,j) = th_phy(i,k,j)*tvcon
   ENDDO
   ENDDO
   ENDDO

   DO j = j_start,j_end
   DO i = i_start,i_end
     govrth(i,j) = g/th_phy(i,1,j)
   ENDDO
   ENDDO

  hflux: SELECT CASE( config_flags%isfflx )
  CASE (0,2) ! with fixed surface heat flux given in the namelist
    heat_flux = config_flags%tke_heat_flux  ! constant heat flux value
    DO j = j_start, j_end
    DO i = i_start, i_end
       cpm = cp * (1. + 0.8 * moist(i,kts,j,P_QV))
       hfx(i,j)= heat_flux*cpm*rho(i,1,j)
    ENDDO
    ENDDO
 
  CASE (1) ! use surface heat flux computed from surface routine
    DO j = j_start, j_end
    DO i = i_start, i_end
       cpm = cp * (1. + 0.8 * moist(i,kts,j,P_QV))
       heat_flux = hfx(i,j)/cpm/rho(i,1,j)
    ENDDO
    ENDDO
 
  CASE DEFAULT
    CALL wrf_error_fatal( 'isfflx value invalid for diff_opt=2' )
  END SELECT hflux

  DO j = j_start, j_end
  DO i = i_start, i_end
     kpbl(i,j)   = 1
     hpbl(i,j)   = zq(i,1,j)
     pblflg(i,j) = .true.
     sfcflg(i,j) = .true.
 
     cpm = cp * (1. + 0.8*moist(i,1,j,P_QV))
     sflux(i,j) = (hfx(i,j)/cpm)/rho(i,1,j)
 
     IF(br(i,j).GT.0.0) sfcflg(i,j) = .false.
  ENDDO
  ENDDO

! get pbl height begin based on theta method
  DO j = j_start, j_end
  DO i = i_start, i_end
        thsfc = thv(i,kts,j) + 0.5
     DO k=kts+1,ktf
        IF(thv(i,k,j).GT.thsfc) THEN
         hpbl(i,j)=za(i,k-1,j)+(thsfc-thv(i,k-1,j))/(max(0.01,thv(i,k,j)-thv(i,k-1,j)))*(za(i,k,j)-za(i,k-1,j))
         kpbl(i,j) = k
         GO TO 119
        ENDIF
     ENDDO
  119 CONTINUE
  ENDDO
  ENDDO

  DO j = j_start, j_end
  DO i = i_start, i_end
     IF(hpbl(i,j).LT.zq(i,2,j)) kpbl(i,j) = 1
     IF(kpbl(i,j).LE.1) pblflg(i,j) = .false.
  ENDDO
  ENDDO

  DO j = j_start, j_end
  DO i = i_start, i_end
     IF(sfcflg(i,j))THEN
       bfx0  = max(sflux(i,j),0.)
       wstar3(i,j) = (govrth(i,j)*bfx0*hpbl(i,j))
       wstar(i,j)  = (wstar3(i,j))**h1
     ELSE
       wstar(i,j)  = 0.
       wstar3(i,j) = 0.
     ENDIF
       ust3 = ust(i,j)**3
       wscale(i,j) = (ust3 + phifac*karman*wstar3(i,j)*0.5)**h1
       wscale(i,j) = MIN(wscale(i,j), ust(i,j)*16.0)
       wscale(i,j) = MAX(wscale(i,j), ust(i,j)/5.0 )
  ENDDO
  ENDDO

  DO j = j_start, j_end
  DO i = i_start, i_end
      delxy = sqrt(dx/msftx(i,j)*dy/msfty(i,j))
      pu1=pu(delxy,hpbl(i,j))
      IF((sfcflg(i,j).and.sflux(i,j).GT.0.0) .and. (hpbl(i,j) .GT. 0)) THEN
!nonlocal momentum flux based on Brown and Grant (1997)
       brint = -sm*ust(i,j)*ust(i,j)*wstar3(i,j)/(hpbl(i,j)*wscale(i,j)**4)
       gamu(i,j) = pu1 * brint*u_phy(i,1,j)/wspd(i,j)
       gamv(i,j) = pu1 * brint*v_phy(i,1,j)/wspd(i,j)
      ELSE
       pblflg(i,j) = .false.
      ENDIF
  ENDDO
  ENDDO

  DO j = j_start,j_end
  DO i = i_start,i_end
     IF(pblflg(i,j))THEN
       k = kpbl(i,j) - 1
       wm3           = wstar3(i,j) + 5. * ust(i,j)**3
       wm2(i,j)      = wm3**h2
       bfxpbl(i,j)   = -0.15*thv(i,1,j)/g*wm3/hpbl(i,j)
       dthv(i,j)     = max(thv(i,k+1,j) - thv(i,k,j),tmin)
       dtheta        = max(th_phy(i,k+1,j) - th_phy(i,k,j),tmin)
 
       du            = u_phy(i,k+1,j)-u_phy(i,k,j)
       dv            = v_phy(i,k+1,j)-v_phy(i,k,j)
 
       we(i,j)       = max(bfxpbl(i,j)/dthv(i,j),-sqrt(wm2(i,j)))
       hfxpbl(i,j)   = we(i,j)*dtheta
       delb          = govrth(i,j)*dthv(i,j)
 
       deltaoh(i,j)  = d1*hpbl(i,j) + d2*wm2(i,j)/delb
       deltaoh(i,j)  = max(ezfac*deltaoh(i,j),hpbl(i,j)-za(i,k,j)-1.)
       deltaoh(i,j)  = min(deltaoh(i,j), hpbl(i,j))
 
       if ((du .ne. 0) .or. (dv .ne. 0)) then
           rigs(i,j) = govrth(i,j)*dthv(i,j)*deltaoh(i,j)/(du**2.+dv**2.)
           rigs(i,j) = max(min(rigs(i,j), rigsmax),rimin)
       else
           rigs(i,j) = rigsmax
       endif
       enlfrac2(i,j) = max(min(wm3/wstar3(i,j)/(1.+cpent/rigs(i,j)),entfmax),entfmin)
       enlfrac2(i,j) = enlfrac2(i,j)*enlfrac
     ENDIF
  ENDDO
  ENDDO

  DO j = j_start, j_end
  DO k = kts, ktf
  DO i = i_start, i_end
       IF(pblflg(i,j))THEN
          entfacmf(i,k,j) = sqrt(((zq(i,k+1,j) - hpbl(i,j))/deltaoh(i,j))**2.)
       ENDIF
  ENDDO
  ENDDO
  ENDDO

  DO j = j_start, j_end
  DO i = i_start, i_end
     IF (pblflg(i,j)) THEN
         deltaoh(i,j) = deltaoh(i,j)/hpbl(i,j)
         delxy = sqrt(dx/msftx(i,j)*dy/msfty(i,j))
         mlfrac        = mltop-deltaoh(i,j)
         ezfrac        = mltop+deltaoh(i,j)
         zfacmf(i,1,j) = min(max((zq(i,2,j)/hpbl(i,j)),zfmin),1.)
         sfcfracn      = max(sfcfracn1,zfacmf(i,1,j))
    !
         sflux0      = (a11+a12*sfcfracn)*sflux(i,j)
         snlflux0    = nlfrac*sflux0
         amf1        = snlflux0/sfcfracn
         amf2        = -snlflux0/(mlfrac-sfcfracn)
         bmf2        = -mlfrac*amf2
         amf3        = snlflux0*enlfrac2(i,j)/deltaoh(i,j)
         bmf3        = -amf3*mlfrac
         hfxpbl(i,j) = amf3+bmf3
         pth1        = pthnl(delxy,hpbl(i,j))
         hfxpbl(i,j) = hfxpbl(i,j)*pth1

        IF(j.EQ.5.AND.(i.EQ.2.OR.i.EQ.3)) THEN
          WRITE(17) REAL(i),REAL(j),hpbl(i,j),REAL(kpbl(i,j)),govrth(i,j),sflux(i,j),wstar3(i,j), &
            wscale(i,j),deltaoh(i,j),rigs(i,j),enlfrac2(i,j),wm2(i,j),dthv(i,j),hfxpbl(i,j),delxy, &
            mlfrac,sfcfracn,sflux0,snlflux0,amf1,amf2,bmf2,amf3,bmf3,pth1
        ENDIF
        DO k = kts, ktf
           zfacmf(i,k,j) = max((zq(i,k+1,j)/hpbl(i,j)),zfmin)
           IF(k.LT.kpbl(i,j)) THEN
             IF(zfacmf(i,k,j).LE.sfcfracn) THEN
               nlflux(i,k,j) =  amf1*zfacmf(i,k,j)
             ELSE IF (zfacmf(i,k,j).LE.mlfrac) THEN
               nlflux(i,k,j) =  amf2*zfacmf(i,k,j)+bmf2
             ENDIF
             nlflux(i,k,j) = nlflux(i,k,j) + hfxpbl(i,j)*exp(-entfacmf(i,k,j))
             nlflux(i,k,j) = nlflux(i,k,j)*pth1
           ENDIF
        ENDDO
    ENDIF
  ENDDO
  ENDDO
CLOSE(17)
END SUBROUTINE nonlocal_flux_probe
END PROGRAM sms_upstream_driver
