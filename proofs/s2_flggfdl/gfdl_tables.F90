! Grid-independent GFDL DATA/computed tables from unchanged WRF routines.
program gfdl_tables
  use module_ra_gfdleta, only: table,o3clim,em1,em1wde,table1,table2,table3, &
    em3,source,dsrce,prgfdl,xduo3n,xdo3n2,xdo3n3,xdo3n4,ab15wd,sko2d,skc1r,sko3r
  implicit none
  integer :: u
  character(len=1024) :: path
  call table
  call o3clim
  call get_command_argument(1,path)
  open(newunit=u,file=trim(path),access='stream',form='unformatted',status='replace')
  write(u) em1,em1wde,table1,table2,table3,em3,source,dsrce,prgfdl, &
    xduo3n,xdo3n2,xdo3n3,xdo3n4,ab15wd,sko2d,skc1r,sko3r
  close(u)
end program
