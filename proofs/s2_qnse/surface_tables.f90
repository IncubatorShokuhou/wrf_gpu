! Export public tables after the unmodified WRF initializer; no copied formulas.
program tables
  use module_sf_qnsesfc
  implicit none
  integer :: lowlyr(1,1), ivg(1,1)
  real :: ust(1,1),z0(1,1),land(1,1),ice(1,1)
  lowlyr=1;ivg=10;ust=.1;z0=.08;land=1.;ice=0.
  call qnsesfcinit(lowlyr,ust,z0,land,ice,ivg,.false.,.false., &
    1,2,1,2,1,17, 1,1,1,1,1,17, 1,1,1,1,1,16)
  if (any(psim1/=psim2).or.any(psih1/=psih2)) stop 4
  if (dzeta1/=dzeta2.or.ztmax1/=ztmax2) stop 5
  open(20,file='tables.bin',access='stream',form='unformatted',status='replace')
  write(20) psim1,psih1,dzeta1,ztmax1
  close(20)
end program
