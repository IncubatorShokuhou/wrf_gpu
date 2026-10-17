! Private visibility shim exposes only these pristine routines; physics unchanged.
program goddard_helpers
  use module_ra_goddard, only: read_lut_nc, opt_cloud, sounding_interp, ozone_interp
  implicit none
  integer, parameter :: nc=16,nz=8
  real :: dz(nc,nz),q(nc,nz,6),re(nc,nz,6)
  real :: tau(nc,nz,11),ssa(nc,nz,11),asy(nc,nz,11)
  real :: tau_lw(nc,nz,10),ssa_lw(nc,nz,10),asy_lw(nc,nz,10)
  real :: rad(nz),lat(7),pressure(nz),sounding(33,6),ozone(nz)
  integer :: i,k,t,j,day(4),c,d
  character(len=32) :: mode
  call get_command_argument(1,mode)
  if(trim(mode)=='equator') then
    call sounding_interp(0.0,100,sounding)
    print *, 'equator sounding sum=',sum(sounding)
    stop
  endif
  rad = (/0.001,0.01,0.015,0.02,10.0,10.001,9999.,10001./)
  lat = (/28.0,46.0,-46.0,90.0,-90.0,22.5,1.0/)
  day = (/1,30,212,365/)
  pressure=(/1100.,1013.,900.,500.,100.,1.,0.001,0.0001/)
  call read_lut_nc(11)
  do i=1,nc
    do k=1,nz
      dz(i,k)=100.+real(11*i+3*k)
      do t=1,6
        re(i,k,t)=rad(k)
        q(i,k,t)=0.
        if(i<=6.and.t==i) q(i,k,t)=1.e-4*real(t+k)
        if(i>=7.and.i<16) q(i,k,t)=1.e-4*real(t+k+i)
      enddo
    enddo
  enddo
  call opt_cloud('sw',nz,11,dz,q,re,tau,ssa,asy,nc)
  call opt_cloud('lw',nz,10,dz,q,re,tau_lw,ssa_lw,asy_lw,nc)
  do i=1,nc
    do k=1,nz
      write(*,'(a,2i5,1p,13e25.16)') 'input',i,k,dz(i,k),q(i,k,:),re(i,k,:)
      do j=1,11
        write(*,'(a,3i5,1p,3e25.16)') 'sw',i,k,j,tau(i,k,j),ssa(i,k,j),asy(i,k,j)
      enddo
      do j=1,10
        write(*,'(a,3i5,1p,3e25.16)') 'lw',i,k,j,tau_lw(i,k,j),ssa_lw(i,k,j),asy_lw(i,k,j)
      enddo
    enddo
  enddo
  do c=1,7
    do d=1,4
      call sounding_interp(lat(c),day(d),sounding)
      call ozone_interp(sounding,nz,pressure,ozone)
      do k=1,33
        write(*,'(a,3i5,1p,7e25.16)') 'climate',c,d,k,lat(c),sounding(k,:)
      enddo
      do k=1,nz
        write(*,'(a,3i5,1p,3e25.16)') 'ozone',c,d,k,lat(c),pressure(k),ozone(k)
      enddo
    enddo
  enddo
end program goddard_helpers
