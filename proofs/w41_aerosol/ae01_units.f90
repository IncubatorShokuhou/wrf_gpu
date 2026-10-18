! AE01 unit oracle: the unchanged pristine WRF V4.7.1 aer_opt=1 Tegen chain (libwrflib.a), CPU, lane-dir build.
!   (1) rrsw_aer rsrtaua/rsrpiza/rsrasya after rrtmg_swinit (swaerpr literals)
!   (2) module_physics_init::aerosol_in on a synthetic lat/lon set (table points, cell interiors, periodic edges)
!   (3) module_radiation_driver::aer_time_int factors via a unit-vector aerodm (aerodt(k) = fact1 at nm, fact2 at np)
!   (4) module_radiation_driver::aer_p_int on synthetic pressure columns (above/below the table, exact table hits)
! Input (stream, native): nlat nlon; lat(nlat); lon(nlon); njul; julian(njul); ncol nz; p(ncol,nz) Pa; pf(ncol,nz+1) Pa;
!                         aerodt(ncol,12,6)
! Output (stream, native): rsrtaua(14,6) rsrpiza(14,6) rsrasya(14,6); aerodm(nlon,12,nlat,12,6); per julian aerodt(12,12);
!                          aerod(ncol,nz,6), totaod(ncol)
program ae01_units
  use module_physics_init, only: aerosol_in
  use module_radiation_driver, only: aer_time_int, aer_p_int
  use module_ra_rrtmg_sw, only: rrtmg_swinit
  use rrsw_aer, only: rsrtaua, rsrpiza, rsrasya
  implicit none
  integer, parameter :: alevsiz = 12, nmon = 12, nsrc = 6
  integer :: nlat, nlon, njul, ncol, nz, i, j, k, m, n
  real, allocatable :: lat(:), lon(:), jul(:), p(:,:), pf(:,:), aerodt_in(:,:,:)
  real, allocatable :: xlat(:,:), xlong(:,:), aerodm(:,:,:,:,:), pina(:)
  real, allocatable :: um(:,:,:,:,:), ut(:,:,:,:)
  real, allocatable :: p3(:,:,:), pf3(:,:,:), at3(:,:,:,:), ad3(:,:,:,:), tot(:,:)
  character(len=256) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', convert='native')
  read(10) nlat, nlon
  allocate(lat(nlat), lon(nlon))
  read(10) lat, lon
  read(10) njul
  allocate(jul(njul))
  read(10) jul
  read(10) ncol, nz
  allocate(p(ncol,nz), pf(ncol,nz+1), aerodt_in(ncol,alevsiz,nsrc))
  read(10) p, pf, aerodt_in
  close(10)

  call rrtmg_swinit(.true., 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1)
  open(20, file=trim(fout), access='stream', form='unformatted', status='replace', convert='native')
  write(20) rsrtaua, rsrpiza, rsrasya

  ! (2) aerosol_in: i = longitude index, j = latitude index (ide/jde one past the last mass point)
  allocate(xlat(nlon,nlat), xlong(nlon,nlat), aerodm(nlon,alevsiz,nlat,nmon,nsrc), pina(alevsiz))
  do j = 1, nlat
    do i = 1, nlon
      xlat(i,j) = lat(j); xlong(i,j) = lon(i)
    end do
  end do
  call aerosol_in(aerodm, pina, alevsiz, nmon, nsrc, xlat, xlong, &
                  1, nlon+1, 1, nlat+1, 1, 2, 1, nlon, 1, nlat, 1, 2, 1, nlon, 1, nlat, 1, 1)
  write(20) pina, aerodm

  ! (3) aer_time_int: unit-vector aerodm -> aerodt(level k) = fact1 (k = nm) + fact2 (k = np)
  allocate(um(1,alevsiz,1,nmon,1), ut(1,alevsiz,1,1))
  um = 0.
  do m = 1, nmon
    um(1,m,1,m,1) = 1.
  end do
  do n = 1, njul
    ut = -999.
    call aer_time_int(1, jul(n), um, ut, alevsiz, nmon, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    write(20) ut(1,1:alevsiz,1,1)
  end do

  ! (4) aer_p_int on ncol single-row columns (i = column, j = 1)
  allocate(p3(ncol,nz+1,1), pf3(ncol,nz+1,1), at3(ncol,alevsiz,1,nsrc), ad3(ncol,nz+1,1,nsrc), tot(ncol,1))
  p3 = 0.; pf3 = 0.; ad3 = -999.
  p3(:,1:nz,1) = p; pf3(:,1:nz+1,1) = pf
  at3(:,:,1,:) = aerodt_in
  call aer_p_int(p3, pina, alevsiz, at3, ad3, nsrc, pf3, tot, &
                 1, ncol+1, 1, 2, 1, nz+1, 1, ncol, 1, 1, 1, nz+1, 1, ncol, 1, 1, 1, nz)
  write(20) ad3(:,1:nz,1,:), tot(:,1)
  close(20)
end program ae01_units
