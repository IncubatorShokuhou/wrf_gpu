program goddard_sw_oracle
  use module_ra_goddard, only: swrad, fast_overcast
  implicit none
  integer,parameter :: n=40,c=16,d=selected_real_kind(15,307)
  real(d) :: p(c,n+1),t(c,n),q(c,n),o(c,n),f(c,n),cosz(c),alb(c)
  real(d) :: tau(c,n,11),ssa(c,n,11),asy(c,n,11),taero(c,n,11),saero(c,n,11),gaero(c,n,11)
  real :: net(c,n+1),down(c,n+1),up(c,n+1),surface(c,4)
  integer :: ict(c),icb(c),k,b,regime
  logical :: mask(c)
  character(len=32) :: arg
  call get_command_argument(1,arg); read(arg,*) regime
  fast_overcast=.true.
  mask=.false.;mask(1)=.true.
  cosz=0.6_d;alb=0.15_d;ict=16;icb=28
  q=0.005_d;o=1.e-7_d;f=0._d
  tau=0._d;ssa=0.99_d;asy=0.85_d
  taero=0._d;saero=0._d;gaero=0._d
  do k=1,n+1
    p(:,k)=10._d+25._d*real(k-1,d)
  enddo
  do k=1,n
    t(:,k)=200._d+100._d*real(k-1,d)/real(n-1,d)
    q(:,k)=q(:,k)*real(k,d)/real(n,d)
  enddo
  if(regime==2.or.regime==3.or.regime==4.or.regime==6.or.regime==8)then
    f(:,12:18)=1._d;tau(:,12:18,:)=2._d
  endif
  if(regime==3)cosz=0.00011_d
  if(regime==4)alb=0.85_d
  if(regime==5)then
    q=0._d;cosz=0.95_d
  endif
  if(regime==6)then
    f(:,30:37)=1._d;tau(:,30:37,:)=30._d
    cosz=0.3_d
  endif
  if(regime==7.or.regime==8)then
    do k=10,30
      do b=1,11
        taero(:,k,b)=0.01_d*real(b,d)
        saero(:,k,b)=0.85_d
        gaero(:,k,b)=0.6_d
      enddo
    enddo
  endif
  call swrad(n,cosz,p,t,q,o,f,ict,icb,tau,ssa,asy,taero,saero,gaero, &
             alb,alb,alb,alb,net,down,up,surface,mask,1)
  write(*,'(a,i4,1p,2e25.16)') 'scalar',icb(1),cosz(1),alb(1)
  write(*,'(a,1p,4e25.16)') 'surface',surface(1,:)
  do k=1,n
    write(*,'(a,i4,1p,4e25.16)') 'layer',k,t(1,k),q(1,k),o(1,k),f(1,k)
    do b=1,11
      write(*,'(a,2i4,1p,6e25.16)') 'optics',k,b,tau(1,k,b),ssa(1,k,b),asy(1,k,b),taero(1,k,b),saero(1,k,b),gaero(1,k,b)
    enddo
  enddo
  do k=1,n+1
    write(*,'(a,i4,1p,4e25.16)') 'interface',k,p(1,k),net(1,k),down(1,k),up(1,k)
  enddo
end program goddard_sw_oracle
