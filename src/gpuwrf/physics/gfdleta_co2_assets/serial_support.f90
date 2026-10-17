! Single-process infrastructure for the native initialization algorithm.
! Every read occurs on this sole monitor; no broadcast is needed.
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_bytes(a,n)
  integer,intent(in) :: n
  integer(kind=1) :: a(n)
end subroutine
subroutine wrf_error_fatal(message)
  character(len=*),intent(in) :: message
  write(0,*) 'GFDL CO2 initialization: ',message
  stop 2
end subroutine
