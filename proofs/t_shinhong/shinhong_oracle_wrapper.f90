! Thin public-call wrapper around the UNMODIFIED WRF 4.7.1 module_bl_shinhong.F
! (lane t-shinhong). Mirrors the module_pbl_driver.F:1301-1329 call: qc/qi are
! passed as QC3D/QI3D with RQCBLTEN/RQIBLTEN present, CORF=f, CTOPO/CTOPO2,
! WRF module_model_constants values (cp=7*r_d/2, r_v=461.6, ep_1=r_v/r_d-1).
! One column, n mass levels; profiles(:,k) columns 1..11 =
!   u v t qv p p8w pi dz8w tke qc qi   (p8w uses n+1 rows, others n)
! surface(1..19) = psfc znt ust hfx qfx wspd br psim psih dt xland u10 v10
!                  dx dy corf ctopo ctopo2 hpbl_in
! result(:,j), j = tke el exch_h rublten rvblten rthblten rqvblten rqcblten
!                  rqiblten scalars(hpbl,kpbl,wstar,delta,u10,v10,regime)
subroutine probe(n, mode, profiles, surface, result) bind(C)
use iso_c_binding
use module_bl_shinhong
implicit none
integer(c_int),value :: n, mode
real(c_float),intent(in) :: profiles(n+1,11), surface(19)
real(c_float),intent(out) :: result(n+1,10)
real :: u(1,n+1,1),v(1,n+1,1),t(1,n+1,1),q(1,n+1,1),p(1,n+1,1)
real :: pf(1,n+1,1),pi(1,n+1,1),dz(1,n+1,1),tk(1,n+1,1),el(1,n+1,1)
real :: qc(1,n+1,1),qi(1,n+1,1),th(1,n+1,1)
real :: ut(1,n+1,1),vt(1,n+1,1),tt(1,n+1,1),qt(1,n+1,1),qct(1,n+1,1),qit(1,n+1,1),kh(1,n+1,1)
real :: ps(1,1),zn(1,1),us(1,1),hp(1,1),pm(1,1),ph(1,1),xl(1,1),hf(1,1),qf(1,1)
real :: ws(1,1),br(1,1),u10(1,1),v10(1,1),cf(1,1),topo(1,1),topo2(1,1)
real :: wstar(1,1),delta(1,1),regime(1,1)
real :: znu(n+1),znw(n+1)
real, parameter :: r_d=287., r_v=461.6, cp=7.*r_d/2., g=9.81
integer :: kp(1,1)
u(1,:,1)=profiles(:,1);v(1,:,1)=profiles(:,2);t(1,:,1)=profiles(:,3)
q(1,:,1)=profiles(:,4);p(1,:,1)=profiles(:,5);pf(1,:,1)=profiles(:,6)
pi(1,:,1)=profiles(:,7);dz(1,:,1)=profiles(:,8);tk(1,:,1)=profiles(:,9)
qc(1,:,1)=profiles(:,10);qi(1,:,1)=profiles(:,11)
th=t/pi
! Tendencies are INTENT(INOUT) accumulators zeroed by WRF before the driver.
ut=0.;vt=0.;tt=0.;qt=0.;qct=0.;qit=0.;kh=0.;el=0.
ps=surface(1);zn=surface(2);us=surface(3);hf=surface(4);qf=surface(5)
ws=surface(6);br=surface(7);pm=surface(8);ph=surface(9)
xl=surface(11);u10=surface(12);v10=surface(13);cf=surface(16)
topo=surface(17);topo2=surface(18);hp=surface(19)
wstar=0.;delta=0.;regime=0.;znu=0.;znw=0.;kp=0
call shinhong(u3d=u,v3d=v,th3d=th,t3d=t,qv3d=q,qc3d=qc,qi3d=qi,p3d=p,p3di=pf,pi3d=pi, &
 rublten=ut,rvblten=vt,rthblten=tt,rqvblten=qt,rqcblten=qct,rqiblten=qit,flag_qi=.true., &
 cp=cp,g=g,rovcp=r_d/cp,rd=r_d,rovg=r_d/g,ep1=r_v/r_d-1.,ep2=r_d/r_v,karman=0.4,xlv=2.5e6,rv=r_v, &
 dz8w=dz,psfc=ps,znu=znu,znw=znw,p_top=5000.,znt=zn,ust=us,hpbl=hp,psim=pm,psih=ph,xland=xl,hfx=hf,qfx=qf,wspd=ws,br=br, &
 dt=surface(10),kpbl2d=kp,exch_h=kh,u10=u10,v10=v10,shinhong_tke_diag=mode,tke_pbl=tk,el_pbl=el,corf=cf, &
 dx=surface(14),dy=surface(15), &
 ids=1,ide=2,jds=1,jde=2,kds=1,kde=n+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=n+1, &
 its=1,ite=1,jts=1,jte=1,kts=1,kte=n,ctopo=topo,ctopo2=topo2,wstar=wstar,delta=delta,regime=regime)
result=0.
result(:,1)=tk(1,:,1);result(:,2)=el(1,:,1);result(:,3)=kh(1,:,1)
result(:,4)=ut(1,:,1);result(:,5)=vt(1,:,1);result(:,6)=tt(1,:,1);result(:,7)=qt(1,:,1)
result(:,8)=qct(1,:,1);result(:,9)=qit(1,:,1)
result(1,10)=hp(1,1);result(2,10)=real(kp(1,1));result(3,10)=wstar(1,1);result(4,10)=delta(1,1)
result(5,10)=u10(1,1);result(6,10)=v10(1,1);result(7,10)=regime(1,1)
end subroutine
