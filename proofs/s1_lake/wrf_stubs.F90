! Only services external to the pristine lake body. REAL constants are copied
! literally from WRF share/module_model_constants.F:19,20,31 (E83).
module module_model_constants
  implicit none
  real, parameter :: r_d=287., cp=7.*r_d/2., rcp=r_d/cp
end module
module module_wrf_error
contains
  subroutine wrf_message(message)
    character(*), intent(in) :: message
    ! WRF diagnostics must not enter the numerical stdout protocol.
    write(0,'(A)') trim(message)
  end subroutine
  subroutine wrf_debug(level,message)
    integer, intent(in) :: level
    character(*), intent(in) :: message
  end subroutine
  subroutine wrf_error_fatal(message)
    character(*), intent(in) :: message
    write(0,'(A)') trim(message)
    stop 2
  end subroutine
end module
