! Only serial WRF infrastructure; physical routines/constants remain unchanged.
module module_configure
  type grid_config_rec_type
    integer :: co2tf = 1
  end type
end module
subroutine wrf_error_fatal(message)
  character(len=*), intent(in) :: message
  write(0,*) 'WRF FATAL: ', message
  stop 2
end subroutine
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function
subroutine wrf_dm_bcast_bytes(a,n)
  integer, intent(in) :: n
  integer(kind=1) :: a(n)
end subroutine
subroutine wrf_dm_bcast_real(a,n)
  integer, intent(in) :: n
  real :: a(n)
end subroutine
subroutine wrf_debug(level,message)
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine
