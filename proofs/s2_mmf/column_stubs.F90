subroutine COMPUTE_VIC_SURFRUNOFF()
 stop 99
end subroutine

subroutine COMPUTE_XAJ_SURFRUNOFF()
 stop 99
end subroutine

subroutine DYNAMIC_VIC()
 stop 99
end subroutine

subroutine TILE_DRAIN()
 stop 99
end subroutine

subroutine TILE_HOOGHOUDT()
 stop 99
end subroutine

subroutine wrf_error_fatal(message)
character(len=*)::message
print *,trim(message)
stop 1
end subroutine
subroutine wrf_message(message)
character(len=*)::message
print *,trim(message)
end subroutine
