! Exact public goddardrad caller operands, REAL4 as radiation_driver supplies.
program goddard_wrapper_oracle
  use module_ra_goddard, only:init_goddardrad,goddardrad
  implicit none
  integer,parameter :: n=12
  real :: t(1,n+1,1),p(1,n+1,1),pw(1,n+1,1),tw(1,n+1,1),rho(1,n+1,1),dz(1,n+1,1),pi(1,n+1,1),cf(1,n+1,1)
  real :: qv(1,n+1,1),qc(1,n+1,1),qi(1,n+1,1),qr(1,n+1,1),qs(1,n+1,1),qg(1,n+1,1),qh(1,n+1,1)
  real :: rc(1,n+1,1),ri(1,n+1,1),rs(1,n+1,1),rz(1,n+1,1)
  real :: lat(1,1),lon(1,1),alb(1,1),em(1,1),tsk(1,1),cosz(1,1),obscur(1,1)
  real :: rate(1,n+1,1),flux(1,1),erbe(1,1,8),surface(1,1,4),cod(1,1),ctop(1,1),dir(1,1),dni(1,1),dif(1,1)
  integer :: k,regime
  character(len=32)::arg
  character(len=2)::part
  call get_command_argument(1,arg);read(arg,*)regime
  lat=28.;lon=-16.;alb=.15;em=.98;tsk=300.;cosz=.6;obscur=0.
  if(regime==3)cosz=-.1
  if(regime==4)then
    lat=-46.;cosz=.1;alb=.8;tsk=270.
  endif
  do k=1,n+1
    pw(1,k,1)=101000.-100000.*real(k-1)/real(n)
    tw(1,k,1)=300.-80.*real(k-1)/real(n)
  enddo
  qv=0.;qc=0.;qi=0.;qr=0.;qs=0.;qg=0.;qh=0.;cf=0.
  t=0.;p=0.;rho=0.;dz=0.;pi=1.
  rc=10.;ri=50.;rs=100.;rz=0.
  do k=1,n
    t(1,k,1)=.5*(tw(1,k,1)+tw(1,k+1,1))
    p(1,k,1)=.5*(pw(1,k,1)+pw(1,k+1,1))
    rho(1,k,1)=p(1,k,1)/(287.*t(1,k,1))
    dz(1,k,1)=1000.
    pi(1,k,1)=(p(1,k,1)/1.e5)**(287./1004.5)
    qv(1,k,1)=.008*(p(1,k,1)/1.e5)**2
    if(regime>=2.and.k>=4.and.k<=8)then
      cf(1,k,1)=1.;qc(1,k,1)=1.e-4;qi(1,k,1)=2.e-5
      qs(1,k,1)=1.e-5;qr(1,k,1)=2.e-5;qg(1,k,1)=1.e-6
    endif
  enddo
  call init_goddardrad(.true.,1,2,1,2,1,n+1,1,1,1,1,1,n+1,1,1,1,1,1,n)
  do k=1,n
    write(*,'(a,i4,1p,17e25.16)')'input',k,t(1,k,1),p(1,k,1),qv(1,k,1),qi(1,k,1),qc(1,k,1),qr(1,k,1),qs(1,k,1),qg(1,k,1), &
      rho(1,k,1),dz(1,k,1),pi(1,k,1),cf(1,k,1),ri(1,k,1),rc(1,k,1),rz(1,k,1),rs(1,k,1),rz(1,k,1)
  enddo
  do k=1,n+1
    write(*,'(a,i4,1p,2e25.16)')'interface',k,pw(1,k,1),tw(1,k,1)
  enddo
  write(*,'(a,1p,6e25.16)')'scalar',lat(1,1),cosz(1,1),1361.,alb(1,1),em(1,1),tsk(1,1)
  part='lw'
  call compute
  write(*,'(a,1p,4e25.16)')'lwflux',erbe(1,1,1:4)
  do k=1,n
    write(*,'(a,i4,1p,e25.16)')'lw',k,rate(1,k,1)
  enddo
  part='sw'
  call compute
  write(*,'(a,1p,4e25.16)')'swflux',erbe(1,1,5:8)
  write(*,'(a,1p,5e25.16)')'swextra',cod(1,1),ctop(1,1),dir(1,1),dni(1,1),dif(1,1)
  do k=1,n
    write(*,'(a,i4,1p,e25.16)')'sw',k,rate(1,k,1)
  enddo
contains
subroutine compute
  rate=0.;flux=0.;erbe=0.;surface=0.;cod=0.;ctop=0.;dir=0.;dni=0.;dif=0.
  call goddardrad(sw_or_lw=part,dx=3000.,rthraten=rate,gsf=flux,xlat=lat,xlong=lon,dz8w=dz,t8w=tw,rho_phy=rho, &
    alb=alb,emiss=em,tsk=tsk,t3d=t,qv=qv,qc=qc,qr=qr,qi=qi,qs=qs,qg=qg,qh=qh,p3d=p,p8w3d=pw,pi3d=pi,cldfra3d=cf, &
    gmt=0.,cp=1004.5,g=9.81,julday=100,xtime=0.,declin=0.,solcon=1361.,radfrq=30.,degrad=3.14159265/180., &
    cod2d_out=cod,ctop2d_out=ctop,warm_rain=.false.,f_qv=.true.,f_qc=.true.,f_qr=.true.,f_qi=.true.,f_qs=.true.,f_qg=.true.,f_qh=.false., &
    rec3d=rc,rei3d=ri,rer3d=rz,res3d=rs,reg3d=rz,reh3d=rz,ids=1,ide=2,jds=1,jde=2,kds=1,kde=n+1, &
    ims=1,ime=1,jms=1,jme=1,kms=1,kme=n+1,its=1,ite=1,jts=1,jte=1,kts=1,kte=n, &
    erbe_out=erbe,itimestep=1,dt_in=18.,sflxd=surface,swddir=dir,swddni=dni,swddif=dif,coszen=cosz,obscur=obscur)
end subroutine compute
end program goddard_wrapper_oracle
