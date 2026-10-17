program ferrier_tables
  use module_mp_etanew, only: gpvs,tbpvs,c1xpvs,c2xpvs
  implicit none
  integer :: u
  character(len=1024) :: path
  call gpvs
  call get_command_argument(1,path)
  open(newunit=u,file=trim(path),access='stream',form='unformatted',status='replace')
  write(u) c1xpvs,c2xpvs,tbpvs
  close(u)
end program
