! Inputs at the exact private swrad boundary; fast_overcast matches goddardrad.
program goddard_sw_definedness
  use module_ra_goddard, only: swrad, fast_overcast
  implicit none
  integer,parameter :: n=40,c=16,d=selected_real_kind(15,307)
  real(d) :: p(c,n+1),t(c,n),q(c,n),o(c,n),f(c,n),cosz(c),alb(c)
  real(d) :: tau(c,n,11),ssa(c,n,11),asy(c,n,11),zero(c,n,11)
  real :: net(c,n+1),down(c,n+1),up(c,n+1),surface(c,4)
  integer :: ict(c),icb(c),k,cloudy
  logical :: mask(c)
  character(len=32) :: arg
  call get_command_argument(1,arg)
  read(arg,*) cloudy
  fast_overcast=.true.
  mask=.false.;mask(1)=.true.
  cosz=0.6_d;alb=0.15_d;ict=16;icb=28
  t=270._d;q=0.005_d;o=1.e-7_d;f=0._d
  tau=0._d;ssa=0.99_d;asy=0.85_d;zero=0._d
  do k=1,n+1
    p(:,k)=10._d+25._d*real(k-1,d)
  enddo
  if(cloudy==1)then
    f(:,12:18)=1._d
    tau(:,12:18,:)=2._d
  endif
  call swrad(n,cosz,p,t,q,o,f,ict,icb,tau,ssa,asy,zero,zero,zero, &
             alb,alb,alb,alb,net,down,up,surface,mask,1)
  write(*,'(a,1p,4e25.16)') 'surface=',surface(1,:)
  do k=1,n+1
    write(*,'(i4,1p,3e25.16)') k,net(1,k),down(1,k),up(1,k)
  enddo
end program goddard_sw_definedness
