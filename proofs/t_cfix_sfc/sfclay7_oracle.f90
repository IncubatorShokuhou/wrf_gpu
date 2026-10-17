! Public-call wrapper around the UNMODIFIED WRF 4.7.1 module_sf_pxsfclay.F (Pleim-Xiu, sf_sfclay_physics=7).
! Mirrors module_surface_driver.F:2165-2176 (CALL PXSFCLAY(..., t_phy, th_phy, ..., dx scalar, ..., itimestep)).
! Layout as sfclay91_oracle.f90; inp(:,21) = dx (first column's value used: WRF passes the scalar DX);
! itimestep is an argument. fm/fh/th2/t2/q2 (absent from PXSFCLAY) are returned as 0.
subroutine probe(n, itimestep, inp, out) bind(C)
use iso_c_binding
use module_sf_pxsfclay
implicit none
integer(c_int), value :: n, itimestep
real(c_float), intent(in) :: inp(n,22)
real(c_float), intent(out) :: out(n,30)
include 'wrf_constants.inc'
real :: u(n,1,1),v(n,1,1),t(n,1,1),th(n,1,1),qv(n,1,1),p(n,1,1),dz(n,1,1)
real, dimension(n,1) :: psfc,tsk,xland,mavail,pblh,znt,ust,mol,qsfc,hfx,qfx,zol,regime
real, dimension(n,1) :: chs,chs2,cqs2,cpm,rmol,psim,psih,lh,flhc,flqc,qgh,u10,v10,gz1oz0,wspd,br
real :: dx
logical, save :: init=.false.
if (.not. init) then
  call pxsfclayinit(.true.)
  init=.true.
end if
u(:,1,1)=inp(:,1); v(:,1,1)=inp(:,2); t(:,1,1)=inp(:,3); th(:,1,1)=inp(:,4); qv(:,1,1)=inp(:,5)
p(:,1,1)=inp(:,6); dz(:,1,1)=inp(:,7)
psfc(:,1)=inp(:,8); tsk(:,1)=inp(:,9); xland(:,1)=inp(:,10); mavail(:,1)=inp(:,12)
pblh(:,1)=inp(:,13); znt(:,1)=inp(:,14); ust(:,1)=inp(:,15); mol(:,1)=inp(:,16); qsfc(:,1)=inp(:,17)
hfx(:,1)=inp(:,18); qfx(:,1)=inp(:,19); zol(:,1)=inp(:,20); dx=inp(1,21); regime(:,1)=inp(:,22)
chs=0.;chs2=0.;cqs2=0.;cpm=0.;rmol=0.;psim=0.;psih=0.;lh=0.;flhc=0.;flqc=0.;qgh=0.
u10=0.;v10=0.;gz1oz0=0.;wspd=0.;br=0.
call pxsfclay(u,v,t,th,qv,p,dz,cp,g,rcp,r_d,xlv,psfc,chs,chs2,cqs2,cpm, &
     znt,ust,pblh,mavail,zol,mol,regime,psim,psih, &
     xland,hfx,qfx,lh,tsk,flhc,flqc,qgh,qsfc,rmol, &
     u10,v10,gz1oz0,wspd,br,1,dx, &
     svp1,svp2,svp3,svpt0,ep_1,ep_2,karman,itimestep, &
     1,n,1,1,1,1, 1,n,1,1,1,1, 1,n,1,1,1,1)
out=0.
out(:,1)=ust(:,1); out(:,2)=znt(:,1); out(:,3)=mol(:,1); out(:,4)=zol(:,1); out(:,5)=rmol(:,1)
out(:,6)=regime(:,1); out(:,7)=psim(:,1); out(:,8)=psih(:,1)
out(:,11)=qsfc(:,1); out(:,12)=qgh(:,1); out(:,13)=cpm(:,1); out(:,14)=hfx(:,1); out(:,15)=qfx(:,1)
out(:,16)=lh(:,1); out(:,17)=flhc(:,1); out(:,18)=flqc(:,1); out(:,19)=chs(:,1); out(:,20)=chs2(:,1)
out(:,21)=cqs2(:,1); out(:,22)=u10(:,1); out(:,23)=v10(:,1)
out(:,27)=gz1oz0(:,1); out(:,28)=wspd(:,1); out(:,29)=br(:,1)
end subroutine probe
