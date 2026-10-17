! Public-call wrapper around the UNMODIFIED WRF 4.7.1 module_sf_sfclay.F (old MM5, sf_sfclay_physics=91).
! Mirrors module_surface_driver.F:2064-2077 (CALL SFCLAY with cp,g,rcp,r_d,xlv,...,ep_1,ep_2,karman,eomeg,
! stbolt,P1000mb,lakemask, ustm,ck,cka,cd,cda,isftcflx=0,iz0tlnd=0,scm_force_flux=0). Lane t-cfix-sfc.
! inp(n,22): u v t_phy th_phy qv p_phy dz8w psfc tsk xland lakemask mavail pblh znt ust mol qsfc hfx qfx zol dx regime
! out(n,30): ust znt mol zol rmol regime psim psih fm fh qsfc qgh cpm hfx qfx lh flhc flqc chs chs2 cqs2
!            u10 v10 th2 t2 q2 gz1oz0 wspd br (30 unused)
subroutine probe(n, inp, out) bind(C)
use iso_c_binding
use module_sf_sfclay
implicit none
integer(c_int), value :: n
real(c_float), intent(in) :: inp(n,22)
real(c_float), intent(out) :: out(n,30)
include 'wrf_constants.inc'
real :: u(n,1,1),v(n,1,1),t(n,1,1),qv(n,1,1),p(n,1,1),dz(n,1,1)
real, dimension(n,1) :: psfc,tsk,xland,lake,mavail,pblh,znt,ust,mol,qsfc,hfx,qfx,zol,dx,regime
real, dimension(n,1) :: chs,chs2,cqs2,cpm,rmol,psim,psih,fm,fh,lh,flhc,flqc,qgh,u10,v10,th2,t2,q2
real, dimension(n,1) :: gz1oz0,wspd,br,ustm,ck,cka,cd,cda
logical, save :: init=.false.
if (.not. init) then
  call sfclayinit(.true.)
  init=.true.
end if
u(:,1,1)=inp(:,1); v(:,1,1)=inp(:,2); t(:,1,1)=inp(:,3); qv(:,1,1)=inp(:,5); p(:,1,1)=inp(:,6); dz(:,1,1)=inp(:,7)
psfc(:,1)=inp(:,8); tsk(:,1)=inp(:,9); xland(:,1)=inp(:,10); lake(:,1)=inp(:,11); mavail(:,1)=inp(:,12)
pblh(:,1)=inp(:,13); znt(:,1)=inp(:,14); ust(:,1)=inp(:,15); mol(:,1)=inp(:,16); qsfc(:,1)=inp(:,17)
hfx(:,1)=inp(:,18); qfx(:,1)=inp(:,19); zol(:,1)=inp(:,20); dx(:,1)=inp(:,21); regime(:,1)=inp(:,22)
chs=0.;chs2=0.;cqs2=0.;cpm=0.;rmol=0.;psim=0.;psih=0.;fm=0.;fh=0.;lh=0.;flhc=0.;flqc=0.;qgh=0.
u10=0.;v10=0.;th2=0.;t2=0.;q2=0.;gz1oz0=0.;wspd=0.;br=0.;ustm=0.;ck=0.;cka=0.;cd=0.;cda=0.
call sfclay(u,v,t,qv,p,dz,cp,g,rcp,r_d,xlv,psfc,chs,chs2,cqs2,cpm, &
     znt,ust,pblh,mavail,zol,mol,regime,psim,psih,fm,fh, &
     xland,hfx,qfx,lh,tsk,flhc,flqc,qgh,qsfc,rmol, &
     u10,v10,th2,t2,q2,gz1oz0,wspd,br,1,dx, &
     svp1,svp2,svp3,svpt0,ep_1,ep_2,karman,eomeg,stbolt,p1000mb,lake, &
     1,n,1,1,1,1, 1,n,1,1,1,1, 1,n,1,1,1,1, &
     ustm,ck,cka,cd,cda,0,0,0)
out=0.
out(:,1)=ust(:,1); out(:,2)=znt(:,1); out(:,3)=mol(:,1); out(:,4)=zol(:,1); out(:,5)=rmol(:,1)
out(:,6)=regime(:,1); out(:,7)=psim(:,1); out(:,8)=psih(:,1); out(:,9)=fm(:,1); out(:,10)=fh(:,1)
out(:,11)=qsfc(:,1); out(:,12)=qgh(:,1); out(:,13)=cpm(:,1); out(:,14)=hfx(:,1); out(:,15)=qfx(:,1)
out(:,16)=lh(:,1); out(:,17)=flhc(:,1); out(:,18)=flqc(:,1); out(:,19)=chs(:,1); out(:,20)=chs2(:,1)
out(:,21)=cqs2(:,1); out(:,22)=u10(:,1); out(:,23)=v10(:,1); out(:,24)=th2(:,1); out(:,25)=t2(:,1)
out(:,26)=q2(:,1); out(:,27)=gz1oz0(:,1); out(:,28)=wspd(:,1); out(:,29)=br(:,1)
end subroutine probe
