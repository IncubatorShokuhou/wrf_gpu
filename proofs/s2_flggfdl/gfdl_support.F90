! Calls the unchanged native support functions, independently of the Python port.
program gfdl_support
  use module_ra_gfdleta, only: zenith, gausin, fpvs_new, gfdletainit
  use module_configure, only: grid_config_rec_type
  implicit none
  integer :: i,j,u,idat(3),hour0,year,month,day,nc,nz,input_year
  real :: ptop,gmt,radt,dt,emiss(1,1)
  real,allocatable :: znw(:),znu(:)
  type(grid_config_rec_type) :: config
  real :: pi,x,garea(311),temps(161),sat(161)
  real :: times,lat(1,1),lon(1,1),czen(1,1),dayi,hour
  character(len=1024) :: input,output
  pi=acos(-1.)
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u) nc,nz,ptop,gmt,radt,dt,input_year
  allocate(znw(nz+1),znu(nz+1));read(u) znw,znu;close(u)
  call gfdletainit(emiss,znw,znu,ptop/1000.,input_year,2,15,gmt,config,.true., &
    1,2,1,2,1,nz+1, 1,1,1,1,1,nz+1, 1,1,1,1,1,nz)
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
  do i=1,311
    x=real(i-1)*.01
    garea(i)=gausin(x)
  enddo
  do i=1,161
    temps(i)=180.+real(i-1)
    sat(i)=fpvs_new(temps(i))
  enddo
  write(u) garea,temps,sat
  do i=1,16
    ! ZENITH's DATA MONTH has implicit SAVE: after a leap-year call February
    ! stays 29. Use non-leap years before leap years so each independent date
    ! matches a fresh WRF process with that initial year.
    year=1999+((i-1)/4)*9
    if (i<=12) then
      if (mod(year,4)==0) year=year+1
    else
      year=2008
    endif
    month=1+mod(i-1,12)
    day=1+mod(i*3,27)
    hour0=mod(i*5,24)
    idat=[month,day,year]
    do j=1,7
      times=real((j-1)*43200)
      lat=real((j-4)*29)*pi/180.
      lon=real((j-4)*55)*pi/180.
      call zenith(times,dayi,hour,idat,hour0,lon,lat,czen,1,1,1,1, &
        1,2,1,2,1,2, 1,1,1,1,1,2, 1,1,1,1,1,1)
      write(u) idat,hour0,times,lat,lon,czen,dayi,hour
    enddo
  enddo
  close(u)
end program
