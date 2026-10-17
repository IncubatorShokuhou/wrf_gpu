! Exact READ/FORMAT boundary from share/wrf_fddaobs_in.F (no QC/preprocessing).
program decode
  implicit none
  character(1024) :: path,outpath
  character(14) :: date_char
  character(40) :: id,namef
  character(16) :: platform,source
  real :: latitude,longitude,elevation,values(2,9)
  logical :: is_sound,bogus
  integer :: meas_count,nlev,nvar,n,ios
  call get_command_argument(1,path)
  call get_command_argument(2,outpath)
  open(10,file=trim(path),status='old')
  open(11,file=trim(outpath),status='replace')
  do
    read(10,101,iostat=ios) date_char
    if(ios < 0) exit
    if(ios /= 0) stop 1
    read(10,102) latitude,longitude
    read(10,1021) id,namef
    read(10,103) platform,source,elevation,is_sound,bogus,meas_count
    write(11,'(a)') date_char
    write(11,'(3(es24.16,1x))') latitude,longitude,elevation
    write(11,'(a)') id
    write(11,'(a)') namef
    write(11,'(a)') platform
    write(11,'(a)') source
    write(11,*) is_sound,bogus,meas_count
    nlev=1; nvar=9
    if(is_sound) then
      nlev=meas_count; nvar=6
    endif
    do n=1,nlev
      if(is_sound) then
        read(10,104) values(:,1:6)
      else
        read(10,105) values
      endif
      write(11,'(*(es24.16,1x))') values(:,1:nvar)
    enddo
  enddo
  include 'obs_formats.inc'
end program
