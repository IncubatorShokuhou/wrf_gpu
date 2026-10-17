! Public-call wrapper around the UNMODIFIED WRF 4.7.1 module_sf_gfs.F (NCEP GFS, sf_sfclay_physics=3).
! Mirrors module_surface_driver.F:2344-2353 (CALL SF_GFS(u,v,t_phy,qv_curr,p_phy,CP,RCP,R_d,XLV,PSFC,...,
! EP_1,EP_2,KARMAN,itimestep)). Layout as sfclay91_oracle.f90; outputs absent from SF_GFS are 0
! (col 9/10 carry PSIM/PSIH duplicates are NOT used; mol/zol/rmol/regime/fm/fh/th2/t2/q2 = 0).
subroutine probe(n, itimestep, inp, out) bind(C)
use iso_c_binding
use module_sf_gfs
use module_gfs_funcphys, only: gfuncphys
implicit none
integer(c_int), value :: n, itimestep
real(c_float), intent(in) :: inp(n,22)
real(c_float), intent(out) :: out(n,30)
include 'wrf_constants.inc'
real :: u(n,1,1),v(n,1,1),t(n,1,1),qv(n,1,1),p(n,1,1)
real, dimension(n,1) :: psfc,tsk,xland,znt,ust
real, dimension(n,1) :: br,chs,chs2,cpm,cqs2,flhc,flqc,gz1oz0,hfx,lh,psim,psih,qfx,qgh,qsfc,u10,v10,wspd
logical, save :: init=.false.
if (.not. init) then
  call gfuncphys()
  init=.true.
end if
u(:,1,1)=inp(:,1); v(:,1,1)=inp(:,2); t(:,1,1)=inp(:,3); qv(:,1,1)=inp(:,5); p(:,1,1)=inp(:,6)
psfc(:,1)=inp(:,8); tsk(:,1)=inp(:,9); xland(:,1)=inp(:,10); znt(:,1)=inp(:,14); ust(:,1)=inp(:,15)
br=0.;chs=0.;chs2=0.;cpm=0.;cqs2=0.;flhc=0.;flqc=0.;gz1oz0=0.;hfx=0.;lh=0.;psim=0.;psih=0.
qfx=0.;qgh=0.;qsfc=0.;u10=0.;v10=0.;wspd=0.
call sf_gfs(u,v,t,qv,p,cp,rcp,r_d,xlv,psfc,chs,chs2,cqs2,cpm, &
     znt,ust,psim,psih,xland,hfx,qfx,lh,tsk,flhc,flqc, &
     qgh,qsfc,u10,v10,gz1oz0,wspd,br,1,ep_1,ep_2,karman,itimestep, &
     1,n,1,1,1,1, 1,n,1,1,1,1, 1,n,1,1,1,1)
out=0.
out(:,1)=ust(:,1); out(:,2)=znt(:,1); out(:,7)=psim(:,1); out(:,8)=psih(:,1)
out(:,11)=qsfc(:,1); out(:,12)=qgh(:,1); out(:,13)=cpm(:,1); out(:,14)=hfx(:,1); out(:,15)=qfx(:,1)
out(:,16)=lh(:,1); out(:,17)=flhc(:,1); out(:,18)=flqc(:,1); out(:,19)=chs(:,1); out(:,20)=chs2(:,1)
out(:,21)=cqs2(:,1); out(:,22)=u10(:,1); out(:,23)=v10(:,1)
out(:,27)=gz1oz0(:,1); out(:,28)=wspd(:,1); out(:,29)=br(:,1)
end subroutine probe
