! Lane w41-topowind: pristine WRF 4.7.1 start_em CTOPO/CTOPO2 oracle.
! The body is the UNMODIFIED dyn_em/start_em.F lines 1535-1628 (IF (.NOT. restart)
! slope/lap_hgt/ctopo block), extracted verbatim by build_ctopo_oracle.py into
! start_em_ctopo_block.inc (sha-checked) and INCLUDEd here.  This wrapper only
! supplies the caller context of start_domain_em: a grid/config_flags derived type
! with the same component names, serial tile bounds (its=ids=1, ite=ide=e_we) and
! the start_em local declarations (start_em.F:113-121: im1..jp1, hx,hy,pi, alpha,vfac).
module ctopo_oracle_types
  implicit none
  type grid_t
    real, allocatable :: ht(:,:), msftx(:,:), msfty(:,:), toposlpx(:,:), toposlpy(:,:), lap_hgt(:,:), &
                         slope(:,:), slp_azi(:,:), cosa(:,:), sina(:,:), ctopo(:,:), ctopo2(:,:), &
                         xland(:,:), var_sso(:,:), var2d(:,:)
    real :: rdx, rdy
  end type
  type flags_t
    logical :: periodic_x, periodic_y, restart
    integer :: topo_wind
  end type
end module

subroutine ctopo_probe(nx, ny, topo_wind, iperx, ipery, ht_in, xland_in, var_sso_in, var2d_in, msftx_in, &
                       msfty_in, cosa_in, sina_in, rdx_in, rdy_in, ctopo_out, ctopo2_out, lap_out) bind(C, name='ctopo_probe')
  use iso_c_binding
  use ctopo_oracle_types
  implicit none
  integer(c_int), value :: nx, ny, topo_wind, iperx, ipery
  real(c_float), intent(in) :: ht_in(nx,ny), xland_in(nx,ny), var_sso_in(nx,ny), var2d_in(nx,ny), &
                               msftx_in(nx,ny), msfty_in(nx,ny), cosa_in(nx,ny), sina_in(nx,ny)
  real(c_float), value :: rdx_in, rdy_in
  real(c_float), intent(out) :: ctopo_out(nx,ny), ctopo2_out(nx,ny), lap_out(nx,ny)
  type(grid_t) :: grid
  type(flags_t) :: config_flags
  integer :: ids, ide, jds, jde, its, ite, jts, jte, ims, ime, jms, jme, i, j
  integer :: im1, ip1, jm1, jp1
  real :: hx, hy, pi
  real :: alpha, vfac
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1
  its = ids; ite = ide; jts = jds; jte = jde
  ims = 0; ime = nx + 1; jms = 0; jme = ny + 1
  allocate(grid%ht(ims:ime,jms:jme), grid%msftx(ims:ime,jms:jme), grid%msfty(ims:ime,jms:jme), &
           grid%toposlpx(ims:ime,jms:jme), grid%toposlpy(ims:ime,jms:jme), grid%lap_hgt(ims:ime,jms:jme), &
           grid%slope(ims:ime,jms:jme), grid%slp_azi(ims:ime,jms:jme), grid%cosa(ims:ime,jms:jme), &
           grid%sina(ims:ime,jms:jme), grid%ctopo(ims:ime,jms:jme), grid%ctopo2(ims:ime,jms:jme), &
           grid%xland(ims:ime,jms:jme), grid%var_sso(ims:ime,jms:jme), grid%var2d(ims:ime,jms:jme))
  grid%ht = 0.; grid%msftx = 1.; grid%msfty = 1.; grid%cosa = 1.; grid%sina = 0.
  grid%xland = 2.; grid%var_sso = 0.; grid%var2d = 0.
  grid%ht(1:nx,1:ny) = ht_in; grid%xland(1:nx,1:ny) = xland_in
  grid%var_sso(1:nx,1:ny) = var_sso_in; grid%var2d(1:nx,1:ny) = var2d_in
  grid%msftx(1:nx,1:ny) = msftx_in; grid%msfty(1:nx,1:ny) = msfty_in
  grid%cosa(1:nx,1:ny) = cosa_in; grid%sina(1:nx,1:ny) = sina_in
  ! periodic halo (filled by WRF halo exchange in the real model)
  grid%ht(0,1:ny) = ht_in(nx,:); grid%ht(nx+1,1:ny) = ht_in(1,:)
  grid%ht(1:nx,0) = ht_in(:,ny); grid%ht(1:nx,ny+1) = ht_in(:,1)
  grid%rdx = rdx_in; grid%rdy = rdy_in
  config_flags%periodic_x = iperx /= 0; config_flags%periodic_y = ipery /= 0
  config_flags%restart = .false.; config_flags%topo_wind = topo_wind
  grid%ctopo = -999.; grid%ctopo2 = -999.; grid%lap_hgt = -999.
include 'start_em_ctopo_block.inc'
  ctopo_out = grid%ctopo(1:nx,1:ny); ctopo2_out = grid%ctopo2(1:nx,1:ny); lap_out = grid%lap_hgt(1:nx,1:ny)
end subroutine
