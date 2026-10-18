! Exact gfortran list-directed REAL read of WRF run/aerosol*.formatted (aerosol_in loop order), stream dump.
program read_tegen
  implicit none
  integer, parameter :: latsiz=46, lonsiz=72, alevsiz=12, nm=12, ns=6
  real :: aerodin(lonsiz,alevsiz,latsiz,nm,ns), pina(alevsiz), lat_od(latsiz), lon_od(lonsiz)
  integer :: i,j,k,m,ks
  open(27,file='aerosol_plev.formatted',form='formatted',status='old')
  do k=1,alevsiz; read(27,*) pina(k); end do; close(27)
  open(28,file='aerosol_lat.formatted',form='formatted',status='old')
  do j=1,latsiz; read(28,*) lat_od(j); end do; close(28)
  open(29,file='aerosol_lon.formatted',form='formatted',status='old')
  do j=1,lonsiz; read(29,*) lon_od(j); end do; close(29)
  open(30,file='aerosol.formatted',form='formatted',status='old')
  do ks=1,ns; do m=1,nm; do j=1,latsiz; do k=1,alevsiz; do i=1,lonsiz
    read(30,*) aerodin(i,k,j,m,ks)
  end do; end do; end do; end do; end do
  close(30)
  open(40,file='tegen_gfortran.bin',form='unformatted',access='stream',status='replace')
  write(40) pina, lat_od, lon_od, aerodin
  close(40)
end program
