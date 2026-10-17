! Pristine-WRF grid-FDDA oracle driver (lane s2-sfdda).
!
! Calls the UNMODIFIED pristine phys/module_fdda_psufddagd.F (FDDAGD) and wraps it
! with VERBATIM transcriptions of the WRF caller logic that owns the FDDA tendency
! lifetime (E11: caller wiring is part of the oracle):
!   * auxinput10 read (med_before_solve_io): the record's *_NDG_OLD/NEW overwrite the pair
!   * phys/module_fddagd_driver.F:258-330  itimestep==1 OLD overwrite, fgdtzero, STEPFG
!     reset + recompute, CALL FDDAGD with the driver's tile bounds (I_END=min(i_end,ide-1))
!   * dyn_em/module_big_step_utilities_em.F calculate_full (mut=mub+mu_2) + calc_mu_uv interior
!   * dyn_em/module_em.F:2420-2470 calculate_phy_tend FDDA coupling (in place)
!   * phys/module_physics_addtendc.F phy_fg_ten: add_a2a / add_c2c_u / add_c2c_v (specified)
!   * dyn_em/module_big_step_utilities_em.F calc_mu_uv_1 interior + phy_prep_part2 decoupling
! Build twice: default REAL (WRF RWORDSIZE=4) and -fdefault-real-8 (fp64 savepoints).
! I/O: native stream binary; arrays in Fortran (i,k,j) order.
program fdda_oracle
  use module_fdda_psufddagd
  implicit none
  integer :: nx, ny, nz, ncalls, nrec, nframes
  integer :: if_no_pbl_nudging_uv, if_no_pbl_nudging_t, if_no_pbl_nudging_q
  integer :: if_zfac_uv, k_zfac_uv, if_zfac_t, k_zfac_t, if_zfac_q, k_zfac_q
  integer :: if_ramping, analysis_interval, end_fdda_hour, end_fdda_hour_sfc, stepfg, fgdtzero, ispec
  real :: guv, gt, gq, dtramp_min, dt, dx
  logical :: specified
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte, i_start, i_end, j_start, j_end
  integer :: i, j, k, c, itimestep, secs, rec_idx, frame_idx, days, seconds
  integer :: itf, jtf, ktf, itsu, jtsv, i_startu, j_startv
  integer :: ib, ie, jb, je, kte_c
  real :: xtime, curr_mins2
  real, allocatable :: c1h(:), c2h(:), ht(:,:)
  real, allocatable :: ru_old(:,:,:,:), ru_new(:,:,:,:), rv_old(:,:,:,:), rv_new(:,:,:,:)
  real, allocatable :: rt_old(:,:,:,:), rt_new(:,:,:,:), rq_old(:,:,:,:), rq_new(:,:,:,:)
  real, allocatable :: fu(:,:,:,:), fv(:,:,:,:), fth(:,:,:,:), fqv(:,:,:,:), fzw(:,:,:,:)
  real, allocatable :: fpblh(:,:,:), fmu2(:,:,:), fmub(:,:,:), fmuts(:,:,:)
  ! WRF memory arrays (ims:ime, kms:kme, jms:jme)
  real, allocatable :: u3d(:,:,:), v3d(:,:,:), th3d(:,:,:), t3d(:,:,:), qv3d(:,:,:), p3d(:,:,:), pi3d(:,:,:)
  real, allocatable :: z(:,:,:), z_at_w(:,:,:), dz8w(:,:,:), hfx_fdda(:,:,:)
  real, allocatable :: u_ndg_old(:,:,:), v_ndg_old(:,:,:), t_ndg_old(:,:,:), q_ndg_old(:,:,:)
  real, allocatable :: u_ndg_new(:,:,:), v_ndg_new(:,:,:), t_ndg_new(:,:,:), q_ndg_new(:,:,:)
  real, allocatable :: rundgdten(:,:,:), rvndgdten(:,:,:), rthndgdten(:,:,:), rqvndgdten(:,:,:)
  real, allocatable :: ru_tendf(:,:,:), rv_tendf(:,:,:), t_tendf(:,:,:), qv_tend(:,:,:)
  real, allocatable :: un(:,:,:), vn(:,:,:), tn(:,:,:), qn(:,:,:)
  real, allocatable :: mu_ndg_old(:,:), mu_ndg_new(:,:), rmundgdten(:,:), pblh(:,:), htm(:,:), regime(:,:), znt(:,:)
  real, allocatable :: sda_hfx(:,:), sda_qfx(:,:), sfc2d(:,:)
  real, allocatable :: sr(:,:,:,:,:), sa(:,:,:,:), ft(:,:,:,:), fp(:,:,:,:), freg(:,:,:), fznt(:,:,:)
  real, allocatable :: mu_2(:,:), mub(:,:), mut(:,:), muu(:,:), muv(:,:), muts(:,:), muus(:,:), muvs(:,:)
  integer :: fdda_sfc, fasdas
  real :: guv_sfc, gt_sfc, gq_sfc, rinblw
  integer :: analysis_interval_sfc
  character(len=512) :: inpath, outpath, hfxpath

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, hfxpath)
  if (len_trim(hfxpath)>0) open(12,file=trim(hfxpath),access="stream",form="unformatted",status="replace")
  open(10, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(10) nx, ny, nz, ncalls, nrec, nframes
  read(10) if_no_pbl_nudging_uv, if_no_pbl_nudging_t, if_no_pbl_nudging_q, &
           if_zfac_uv, k_zfac_uv, if_zfac_t, k_zfac_t, if_zfac_q, k_zfac_q, &
           if_ramping, analysis_interval, end_fdda_hour, end_fdda_hour_sfc, stepfg, fgdtzero, ispec
  read(10) guv, gt, gq, dtramp_min, dt, dx
  read(10) fdda_sfc, analysis_interval_sfc
  read(10) guv_sfc, gt_sfc, gq_sfc, rinblw
  fasdas=0
  if(fdda_sfc == 2) fasdas=1
  specified = ispec == 1

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = ids; ime = ide; jms = jds; jme = jde; kms = kds; kme = kde
  ! fddagd_driver tile: I_START=grid%i_start, I_END=min(grid%i_end, ide-1), KTE=min(k_end,kde-1)
  i_start = ids; i_end = min(ide, ide - 1); j_start = jds; j_end = min(jde, jde - 1)
  kts = kds; kte = min(kde, kde - 1)

  allocate(c1h(kms:kme), c2h(kms:kme), ht(nx, ny))
  c1h = 0.; c2h = 0.
  read(10) c1h(1:nz), c2h(1:nz)
  read(10) ht
  allocate(ru_old(nx+1, nz, ny, nrec), ru_new(nx+1, nz, ny, nrec), rv_old(nx, nz, ny+1, nrec), rv_new(nx, nz, ny+1, nrec))
  allocate(rt_old(nx, nz, ny, nrec), rt_new(nx, nz, ny, nrec), rq_old(nx, nz, ny, nrec), rq_new(nx, nz, ny, nrec))
  do c = 1, nrec
    read(10) ru_old(:,:,:,c), ru_new(:,:,:,c), rv_old(:,:,:,c), rv_new(:,:,:,c)
    read(10) rt_old(:,:,:,c), rt_new(:,:,:,c), rq_old(:,:,:,c), rq_new(:,:,:,c)
  end do
  allocate(sr(nx+1,ny+1,5,2,nrec),sa(nx+1,ny+1,5,2))
  read(10) sr
  allocate(ft(nx,nz,ny,nframes),fp(nx,nz,ny,nframes),freg(nx,ny,nframes),fznt(nx,ny,nframes))
  allocate(fu(nx+1, nz, ny, nframes), fv(nx, nz, ny+1, nframes), fth(nx, nz, ny, nframes), fqv(nx, nz, ny, nframes))
  allocate(fzw(nx, nz+1, ny, nframes), fpblh(nx, ny, nframes), fmu2(nx, ny, nframes), fmub(nx, ny, nframes))
  allocate(fmuts(nx, ny, nframes))
  do c = 1, nframes
    read(10) fu(:,:,:,c), fv(:,:,:,c), fth(:,:,:,c), fqv(:,:,:,c), fzw(:,:,:,c)
    read(10) fpblh(:,:,c), fmu2(:,:,c), fmub(:,:,c), fmuts(:,:,c)
    read(10) ft(:,:,:,c),fp(:,:,:,c),freg(:,:,c),fznt(:,:,c)
  end do

  allocate(u3d(ims:ime,kms:kme,jms:jme), v3d(ims:ime,kms:kme,jms:jme), th3d(ims:ime,kms:kme,jms:jme))
  allocate(t3d(ims:ime,kms:kme,jms:jme), qv3d(ims:ime,kms:kme,jms:jme), p3d(ims:ime,kms:kme,jms:jme))
  allocate(pi3d(ims:ime,kms:kme,jms:jme), z(ims:ime,kms:kme,jms:jme), z_at_w(ims:ime,kms:kme,jms:jme))
  allocate(dz8w(ims:ime,kms:kme,jms:jme), hfx_fdda(ims:ime,kms:kme,jms:jme))
  allocate(u_ndg_old(ims:ime,kms:kme,jms:jme), v_ndg_old(ims:ime,kms:kme,jms:jme))
  allocate(t_ndg_old(ims:ime,kms:kme,jms:jme), q_ndg_old(ims:ime,kms:kme,jms:jme))
  allocate(u_ndg_new(ims:ime,kms:kme,jms:jme), v_ndg_new(ims:ime,kms:kme,jms:jme))
  allocate(t_ndg_new(ims:ime,kms:kme,jms:jme), q_ndg_new(ims:ime,kms:kme,jms:jme))
  allocate(rundgdten(ims:ime,kms:kme,jms:jme), rvndgdten(ims:ime,kms:kme,jms:jme))
  allocate(rthndgdten(ims:ime,kms:kme,jms:jme), rqvndgdten(ims:ime,kms:kme,jms:jme))
  allocate(ru_tendf(ims:ime,kms:kme,jms:jme), rv_tendf(ims:ime,kms:kme,jms:jme))
  allocate(t_tendf(ims:ime,kms:kme,jms:jme), qv_tend(ims:ime,kms:kme,jms:jme))
  allocate(un(ims:ime,kms:kme,jms:jme), vn(ims:ime,kms:kme,jms:jme), tn(ims:ime,kms:kme,jms:jme), qn(ims:ime,kms:kme,jms:jme))
  allocate(mu_ndg_old(ims:ime,jms:jme), mu_ndg_new(ims:ime,jms:jme), rmundgdten(ims:ime,jms:jme))
  allocate(pblh(ims:ime,jms:jme), htm(ims:ime,jms:jme), regime(ims:ime,jms:jme), znt(ims:ime,jms:jme))
  allocate(sda_hfx(ims:ime,jms:jme), sda_qfx(ims:ime,jms:jme), sfc2d(ims:ime,jms:jme))
  allocate(mu_2(ims:ime,jms:jme), mub(ims:ime,jms:jme), mut(ims:ime,jms:jme), muu(ims:ime,jms:jme))
  allocate(muv(ims:ime,jms:jme), muts(ims:ime,jms:jme), muus(ims:ime,jms:jme), muvs(ims:ime,jms:jme))

  u3d = 0.; v3d = 0.; th3d = 0.; t3d = 0.; qv3d = 0.; p3d = 0.; pi3d = 0.; z = 0.; z_at_w = 0.; dz8w = 0.
  hfx_fdda = 0.
  u_ndg_old = 0.; v_ndg_old = 0.; t_ndg_old = 0.; q_ndg_old = 0.
  u_ndg_new = 0.; v_ndg_new = 0.; t_ndg_new = 0.; q_ndg_new = 0.
  mu_ndg_old = 0.; mu_ndg_new = 0.; pblh = 0.; htm = 0.; regime = 0.; znt = 0.; sda_hfx = 0.; sda_qfx = 0.; sfc2d = 0.
  mu_2 = 0.; mub = 0.; mut = 0.; muu = 0.; muv = 0.; muts = 0.; muus = 0.; muvs = 0.
  ! fddagdinit (.not. restart): tendencies = 0
  rundgdten = 0.; rvndgdten = 0.; rthndgdten = 0.; rqvndgdten = 0.; rmundgdten = 0.
  htm(1:nx, 1:ny) = ht
  sa=0.

  open(11, file=trim(outpath), access='stream', form='unformatted', status='replace')
  do c = 1, ncalls
    read(10) itimestep, secs, rec_idx, frame_idx
    ! ---- per-step model inputs (frame) ----
    u3d(1:nx+1, 1:nz, 1:ny) = fu(:,:,:,frame_idx)
    v3d(1:nx, 1:nz, 1:ny+1) = fv(:,:,:,frame_idx)
    th3d(1:nx, 1:nz, 1:ny) = fth(:,:,:,frame_idx)
    qv3d(1:nx, 1:nz, 1:ny) = fqv(:,:,:,frame_idx)
    z_at_w(1:nx, 1:nz+1, 1:ny) = fzw(:,:,:,frame_idx)
    pblh(1:nx, 1:ny) = fpblh(:,:,frame_idx)
    mu_2(1:nx, 1:ny) = fmu2(:,:,frame_idx)
    mub(1:nx, 1:ny) = fmub(:,:,frame_idx)
    muts(1:nx, 1:ny) = fmuts(:,:,frame_idx)
    t3d(1:nx,1:nz,1:ny)=ft(:,:,:,frame_idx)
    p3d(1:nx,1:nz,1:ny)=fp(:,:,:,frame_idx)
    regime(1:nx,1:ny)=freg(:,:,frame_idx)
    znt(1:nx,1:ny)=fznt(:,:,frame_idx)
    ! phy_prep:4876, all mass layers use adjacent w-level heights.
    dz8w(:,1:nz,:)=z_at_w(:,2:nz+1,:)-z_at_w(:,1:nz,:)
    z(:,1:nz,:)=0.5*(z_at_w(:,1:nz,:)+z_at_w(:,2:nz+1,:))
    ! ---- clocks (frame/module_domain.F:2186, dyn_em/solve_em.F:339) ----
    days = secs / 86400
    seconds = secs - days * 86400
    xtime = ( REAL( days ) * 24. * 60. ) + ( REAL( seconds ) / 60. )
    curr_mins2 = REAL( DBLE( secs ) / 60.0d0 )
    ! ---- auxinput10 read ----
    if (rec_idx >= 1) then
      sa=sr(:,:,:,:,rec_idx)
      u_ndg_old(1:nx+1, 1:nz, 1:ny) = ru_old(:,:,:,rec_idx)
      u_ndg_new(1:nx+1, 1:nz, 1:ny) = ru_new(:,:,:,rec_idx)
      v_ndg_old(1:nx, 1:nz, 1:ny+1) = rv_old(:,:,:,rec_idx)
      v_ndg_new(1:nx, 1:nz, 1:ny+1) = rv_new(:,:,:,rec_idx)
      t_ndg_old(1:nx, 1:nz, 1:ny) = rt_old(:,:,:,rec_idx)
      t_ndg_new(1:nx, 1:nz, 1:ny) = rt_new(:,:,:,rec_idx)
      q_ndg_old(1:nx, 1:nz, 1:ny) = rq_old(:,:,:,rec_idx)
      q_ndg_new(1:nx, 1:nz, 1:ny) = rq_new(:,:,:,rec_idx)
    end if
    ! ---- fddagd_driver (phys/module_fddagd_driver.F) ----
    IF (itimestep == 1) THEN
      DO j = j_start, j_end
      DO i = i_start, i_end
         DO k = kts, kte
            u_ndg_old(i,k,j) = u3d(i,k,j)
            v_ndg_old(i,k,j) = v3d(i,k,j)
            t_ndg_old(i,k,j) = th3d(i,k,j) - 300.0
            q_ndg_old(i,k,j) = qv3d(i,k,j)
         ENDDO
         mu_ndg_old(i,j) = 0.0
      ENDDO
      ENDDO
    ENDIF
    IF (mod(itimestep-1,STEPFG) .eq. 0 .and. fgdtzero .eq. 1) THEN
      DO j = j_start, j_end
      DO i = i_start, i_end
         DO k = kts, min(kte+1,kde)
            RTHNDGDTEN(I,K,J)=0.
            RUNDGDTEN(I,K,J)=0.
            RVNDGDTEN(I,K,J)=0.
            RQVNDGDTEN(I,K,J)=0.
         ENDDO
         RMUNDGDTEN(I,J)=0.
      ENDDO
      ENDDO
    ENDIF
    IF (itimestep .eq. 1 .or. mod(itimestep,STEPFG) .eq. 0) THEN
      DO j = j_start, j_end
      DO i = i_start, i_end
         DO k = kts, min(kte+1,kde)
            RTHNDGDTEN(I,K,J)=0.
            RUNDGDTEN(I,K,J)=0.
            RVNDGDTEN(I,K,J)=0.
            RQVNDGDTEN(I,K,J)=0.
         ENDDO
         RMUNDGDTEN(I,J)=0.
      ENDDO
      ENDDO
      CALL FDDAGD(itimestep,dx,dt,xtime, curr_mins2, 1, &
           analysis_interval, end_fdda_hour, &
           if_no_pbl_nudging_uv, if_no_pbl_nudging_t, if_no_pbl_nudging_q, &
           if_zfac_uv, k_zfac_uv, if_zfac_t, k_zfac_t, if_zfac_q, k_zfac_q, &
           guv, gt, gq, if_ramping, dtramp_min, &
           fdda_sfc, analysis_interval_sfc, end_fdda_hour_sfc, guv_sfc, gt_sfc, gq_sfc, rinblw, &
           u3d, v3d, th3d, t3d, qv3d, p3d, pi3d, &
           u_ndg_old, v_ndg_old, t_ndg_old, q_ndg_old, mu_ndg_old, &
           u_ndg_new, v_ndg_new, t_ndg_new, q_ndg_new, mu_ndg_new, &
           sa(:,:,1,1),sa(:,:,2,1),sfc2d,sa(:,:,3,1),sa(:,:,4,1),sfc2d,sfc2d,sfc2d,sfc2d,sa(:,:,5,1), &
           sa(:,:,1,2),sa(:,:,2,2),sfc2d,sa(:,:,3,2),sa(:,:,4,2),sfc2d,sfc2d,sfc2d,sfc2d,sa(:,:,5,2), &
           RUNDGDTEN, RVNDGDTEN, RTHNDGDTEN, RQVNDGDTEN, RMUNDGDTEN, &
           fasdas, SDA_HFX, SDA_QFX, HFX_FDDA, dz8w, &
           pblh, htm, regime, znt, z, z_at_w, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           i_start, i_end, j_start, j_end, kts, kte)
    ENDIF
    un = rundgdten; vn = rvndgdten; tn = rthndgdten; qn = rqvndgdten
    ! ---- rk_step_prep: calculate_full + calc_mu_uv (interior faces) ----
    DO j = 1, ny
    DO i = 1, nx
      mut(i,j) = mub(i,j) + mu_2(i,j)
    ENDDO
    ENDDO
    DO j = 1, ny
    DO i = 2, nx
      MUU(i,j) = 0.5*(MU_2(i,j)+MU_2(i-1,j)+MUB(i,j)+MUB(i-1,j))
    ENDDO
    ENDDO
    DO j = 2, ny
    DO i = 1, nx
      MUV(i,j) = 0.5*(MU_2(i,j)+MU_2(i,j-1)+MUB(i,j)+MUB(i,j-1))
    ENDDO
    ENDDO
    ! ---- calculate_phy_tend FDDA block (module_em.F:2424-2470), tile its..min(ite,ide-1) ----
    itf = MIN(nx, ide-1); jtf = MIN(ny, jde-1); ktf = MIN(nz, kde-1)
    itsu = MAX(1, ids+1); jtsv = MAX(1, jds+1)
    DO J=1,jtf
    DO K=1,ktf
    DO I=itsu,itf
       RUNDGDTEN(I,K,J) =(c1h(k)*muu(I,J)+c2h(k))*RUNDGDTEN(I,K,J)
    ENDDO
    ENDDO
    ENDDO
    DO J=jtsv,jtf
    DO K=1,ktf
    DO I=1,itf
       RVNDGDTEN(I,K,J) =(c1h(k)*muv(I,J)+c2h(k))*RVNDGDTEN(I,K,J)
    ENDDO
    ENDDO
    ENDDO
    DO J=1,jtf
    DO K=1,ktf
    DO I=1,itf
       RTHNDGDTEN(I,K,J)=(c1h(k)*mut(I,J)+c2h(k))*RTHNDGDTEN(I,K,J)
    ENDDO
    ENDDO
    ENDDO
    DO J=1,jtf
    DO K=1,ktf
    DO I=1,itf
       RQVNDGDTEN(I,K,J)=(c1h(k)*mut(I,J)+c2h(k))*RQVNDGDTEN(I,K,J)
    ENDDO
    ENDDO
    ENDDO
    ! ---- phy_fg_ten (update_phy_ten tile: i_start..grid%i_end, k_start..k_end) ----
    ru_tendf = 0.; rv_tendf = 0.; t_tendf = 0.; qv_tend = 0.
    kte_c = nz + 1
    ! add_a2a(rt_tendf, RTHNDGDTEN) and add_a2a(moist_tendf(P_QV), RQVNDGDTEN)
    ib = 1; ie = MIN(ide,ide-1); jb = 1; je = MIN(jde,jde-1)
    IF (specified) THEN
      ib = MAX(ids+1,1); ie = MIN(ide-2,ide); jb = MAX(jds+1,1); je = MIN(jde-2,jde)
    ENDIF
    DO j = jb, je
    DO k = 1, MIN(kte_c,kde-1)
    DO i = ib, ie
      t_tendf(i,k,j) = t_tendf(i,k,j) + RTHNDGDTEN(i,k,j)
      qv_tend(i,k,j) = qv_tend(i,k,j) + RQVNDGDTEN(i,k,j)
    ENDDO
    ENDDO
    ENDDO
    ! add_c2c_u
    ib = 1; ie = ide; jb = 1; je = MIN(jde,jde-1)
    IF (specified) THEN
      ib = MAX(ids+1,1); ie = MIN(ide-1,ide); jb = MAX(jds+1,1); je = MIN(jde-2,jde)
    ENDIF
    DO j = jb, je
    DO k = 1, MIN(kte_c,kde-1)
    DO i = ib, ie
      ru_tendf(i,k,j) = ru_tendf(i,k,j) + RUNDGDTEN(i,k,j)
    ENDDO
    ENDDO
    ENDDO
    ! add_c2c_v
    ib = 1; ie = MIN(ide,ide-1); jb = 1; je = jde
    IF (specified) THEN
      ib = MAX(ids+1,1); ie = MIN(ide-2,ide); jb = MAX(jds+1,1); je = MIN(jde-1,jde)
    ENDIF
    DO j = jb, je
    DO k = 1, kte_c
    DO i = ib, ie
      rv_tendf(i,k,j) = rv_tendf(i,k,j) + RVNDGDTEN(i,k,j)
    ENDDO
    ENDDO
    ENDDO
    ! ---- outputs: pair, uncoupled, coupled, owned ----
    write(11) u_ndg_old(1:nx+1,1:nz,1:ny), u_ndg_new(1:nx+1,1:nz,1:ny), v_ndg_old(1:nx,1:nz,1:ny+1)
    write(11) v_ndg_new(1:nx,1:nz,1:ny+1), t_ndg_old(1:nx,1:nz,1:ny), t_ndg_new(1:nx,1:nz,1:ny)
    write(11) q_ndg_old(1:nx,1:nz,1:ny), q_ndg_new(1:nx,1:nz,1:ny)
    write(11) un(1:nx+1,1:nz,1:ny), vn(1:nx,1:nz,1:ny+1), tn(1:nx,1:nz,1:ny), qn(1:nx,1:nz,1:ny)
    write(11) rundgdten(1:nx+1,1:nz,1:ny), rvndgdten(1:nx,1:nz,1:ny+1), rthndgdten(1:nx,1:nz,1:ny), &
              rqvndgdten(1:nx,1:nz,1:ny)
    write(11) ru_tendf(1:nx+1,1:nz,1:ny), rv_tendf(1:nx,1:nz,1:ny+1), t_tendf(1:nx,1:nz,1:ny), qv_tend(1:nx,1:nz,1:ny)
    ! ---- solve_em: calc_mu_uv_1(muts) interior + phy_prep_part2 FDDA decoupling ----
    DO j = 1, ny
    DO i = 2, nx
      MUUS(i,j) = 0.5*(MUTS(i,j)+MUTS(i-1,j))
    ENDDO
    ENDDO
    DO j = 2, ny
    DO i = 1, nx
      MUVS(i,j) = 0.5*(MUTS(i,j)+MUTS(i,j-1))
    ENDDO
    ENDDO
    i_startu = MAX(1, ids+1); j_startv = MAX(1, jds+1)
    DO J=1,MIN(ny,jde-1)
    DO K=1,MIN(nz,kde-1)
    DO I=i_startu,MIN(nx,ide-1)
       RUNDGDTEN(I,K,J) =RUNDGDTEN(I,K,J)/(c1h(k)*muus(I,J)+c2h(k))
    ENDDO
    ENDDO
    ENDDO
    DO J=j_startv,MIN(ny,jde-1)
    DO K=1,MIN(nz,kde-1)
    DO I=1,MIN(nx,ide-1)
       RVNDGDTEN(I,K,J) =RVNDGDTEN(I,K,J)/(c1h(k)*muvs(I,J)+c2h(k))
    ENDDO
    ENDDO
    ENDDO
    DO J=1,MIN(ny,jde-1)
    DO K=1,MIN(nz,kde-1)
    DO I=1,MIN(nx,ide-1)
       RTHNDGDTEN(I,K,J)=RTHNDGDTEN(I,K,J)/(c1h(k)*MUTS(I,J)+c2h(k))
       RQVNDGDTEN(I,K,J)=RQVNDGDTEN(I,K,J)/(c1h(k)*MUTS(I,J)+c2h(k))
    ENDDO
    ENDDO
    ENDDO
    write(11) rundgdten(1:nx+1,1:nz,1:ny), rvndgdten(1:nx,1:nz,1:ny+1), rthndgdten(1:nx,1:nz,1:ny), &
              rqvndgdten(1:nx,1:nz,1:ny)
    write(11) SDA_HFX(1:nx,1:ny), SDA_QFX(1:nx,1:ny)
    if (len_trim(hfxpath)>0) write(12) HFX_FDDA(1:nx,1:nz,1:ny)
  end do
  close(10); close(11)
end program fdda_oracle

subroutine wrf_message(str)
  character(len=*) :: str
end subroutine wrf_message

subroutine wrf_error_fatal(str)
  character(len=*) :: str
  write(0,*) 'FATAL: ', trim(str)
  stop 3
end subroutine wrf_error_fatal

subroutine get_wrf_debug_level(level)
  integer :: level
  level = 0
end subroutine get_wrf_debug_level

logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .false.
end function wrf_dm_on_monitor
