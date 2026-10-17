! Full first_rk_step_part2 SMS operator: pristine upstream, coefficients, RHS, implicit + horizontal diffusion.
! Derived from proofs/o1_smag3d's caller/halo driver. Deliberately stops BEFORE
! phy_bc/nonlocal/vertical_diffusion_implicit; hpbl/dlk supplied independent operands.
! Header km_opt=2 is only the shared input format; execute km_opt=5 after reading.
! No solve_em invocation, so no E104 shared hard-coded dump path is reachable.
PROGRAM sms_full_driver
  USE module_model_constants, ONLY : ep_1, ep_2, karman
  USE module_configure, ONLY : grid_config_rec_type
  USE module_state_description, ONLY : p_qv, p_qc, p_qr, p_qi, p_qs, p_qg, p_qni, p_qnr, param_first_scalar
  USE module_bc, ONLY : set_physical_bc3d, set_physical_bc2d
  USE module_diffusion_em, ONLY : compute_diff_metrics, cal_deform_and_div, calculate_km_kh, &
                                  tke_rhs, phy_bc, vertical_diffusion_implicit, horizontal_diffusion_2, nonlocal_flux, free_atmos_length, meso_length_scale
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
  REAL, ALLOCATABLE :: c1f(:),c2f(:)
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
  p_qv = 2; p_qc = 3; p_qr = 4; p_qi = 5; p_qs = 6; p_qg=7; p_qni=2; p_qnr=3

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
  ALLOCATE(scal(ims:ime,kms:kme,jms:jme,3), scal_t(ims:ime,kms:kme,jms:jme,3)); scal = 0.; scal_t = 0.
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
  ALLOCATE(c1f(kms:kme),c2f(kms:kme))
  READ(10) c1f,c2f
  READ(10) scal(ids:ide-1,1:nz,jds:jde-1,2),scal(ids:ide-1,1:nz,jds:jde-1,3)
  READ(10) ustm(ids:ide-1,jds:jde-1)
  CALL bc3(scal(:,:,:,2),'t');CALL bc3(scal(:,:,:,3),'t')
  hfx_start=hfx;gu_start=gamu;gv_start=gamv
  CLOSE(10)
  cf%km_opt = 5
  ! USTM supplied independently of convective UST, as surface_driver does.

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
  CALL nonlocal_flux(cf,nlflux,gamu,gamv,pblh,kpbl,dx,dy,dt,ust,hfx,qfx,br,ep_1,ep_2,karman, &
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
  CALL phy_bc(cf,div,d11,d22,d33,d12,d13,d23,xkmh,xkmv,xkhh,xkhv,tke,rho, &
              rublten,rvblten,rucuten,rvcuten,rushten,rvshten,gamu,gamv,xkmv_meso, &
              ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ids,ide,jds,jde,kds,kde,its,ite,jts,jte,kts,kte)
  CALL tke_rhs(tke_t,bn2,cf,d11,d22,d33,d12,d13,d23,u,v,w,div,tke,mut,c1h,c2h, &
               th_phy,p_phy,p8w,t8w,z,fnm,fnp,cf1,cf2,cf3,msftx,msfty,xkmh,xkmv,xkhv, &
               rdx,rdy,dx,dy,dt,zx,zy,rdz,rdzw,dn,dnw,cf%mix_isotropic,hfx,qfx,moist(:,:,:,p_qv), &
               ustm,rho,l_diss,nlflux,pblh,dlk, &
               ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  CALL vertical_diffusion_implicit(ru_t,rv_t,rw_t,t_t,tke_t,moist_t,n_moist,chem_t,1,scal_t,3,trac_t,1, &
             u,v,w,dt,thp,u_base,v_base,t_base,qv_base,mut,tke,th_phy,cf,moist,chem,scal,trac, &
             xkmv,xkhv,xkmh,cf%km_opt,fnm,fnp,dn,dnw,rdz,rdzw,hfx,qfx,ustm,rho, &
             nlflux,gamu,gamv,xkmv_meso,l_diss,c1h,c2h,c1f,c2f, &
             ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  CALL horizontal_diffusion_2(t_t,ru_t,rv_t,rw_t,tke_t,moist_t,n_moist,chem_t,1,scal_t,3,trac_t,1, &
             thp,th_phy,tke,cf,d11,d22,d12,d13,d23,nba_mij,1,div,moist,chem,scal,trac, &
             msfux,msfuy,msfvx,msfvy,msftx,msfty,xkmh,xkmv,xkhh,cf%km_opt,rdx,rdy,rdz,rdzw, &
             fnm,fnp,cf1,cf2,cf3,zx,zy,dn,dnw,rho, &
             ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
  OPEN(11, FILE=TRIM(fout), ACCESS='stream', FORM='unformatted', STATUS='replace')
  WRITE(11) ims, ime, jms, jme, kms, kme
  WRITE(11) ru_t,rv_t,rw_t,t_t,tke_t
  WRITE(11) moist_t(:,:,:,2:n_moist),scal_t(:,:,:,2:3)
  WRITE(11) xkmh,xkmv,xkhh,xkhv,xkmv_meso,l_diss,nlflux,dlk,elmin
  WRITE(11) gamu,gamv,pblh,hfx
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
END PROGRAM sms_full_driver
