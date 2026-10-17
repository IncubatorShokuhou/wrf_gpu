! Thin public-call wrapper around the UNMODIFIED WRF 4.7.1 YSU PBL
! (phys/module_bl_ysu.F -> phys/physics_mmm/bl_ysu.F90 bl_ysu_run), lane t-ysuqc.
! Mirrors the module_pbl_driver.F:1208-1251 YSUSCHEME call: QC3D/QI3D with
! RQCBLTEN/RQIBLTEN present, FLAG_QC/FLAG_QI from the microphysics package,
! PSIM=fm PSIH=fhh (selected SF1 solve), CTOPO/CTOPO2 present (start_em.F:1581,
! 1. for topo_wind=0), RTHRATEN, YSU_TOPDOWN_PBLMIX, UOCE/VOCE, no BEP
! (flag_bep=.false.), module_model_constants values (cp=7*r_d/2, r_v=461.6,
! rcp=r_d/cp, rovg=r_d/g, ep_1=r_v/r_d-1, ep_2=r_d/r_v, XLV=2.5e6).
! One column, n mass levels; profiles(:,k) columns 1..11 =
!   u v t qv qc qi p p8w pi dz8w rthraten   (p8w uses n+1 rows, others n)
! surface(1..17) = psfc znt ust psim psih xland hfx qfx wspd br u10 v10
!                  uoce voce ctopo ctopo2 dt
! flags = topdown flag_qc flag_qi
! result(:,j), j = rublten rvblten rthblten rqvblten rqcblten rqiblten
!                  exch_h exch_m scalars(hpbl,kpbl,wstar,delta,u10,v10)
subroutine probe(n, topdown, fqc, fqi, profiles, surface, result) bind(C)
use iso_c_binding
use module_bl_ysu
implicit none
integer(c_int),value :: n, topdown, fqc, fqi
real(c_float),intent(in) :: profiles(n+1,11), surface(17)
real(c_float),intent(out) :: result(n+1,9)
real :: u(1,n+1,1),v(1,n+1,1),t(1,n+1,1),q(1,n+1,1),qc(1,n+1,1),qi(1,n+1,1)
real :: p(1,n+1,1),pf(1,n+1,1),pi(1,n+1,1),dz(1,n+1,1),rthr(1,n+1,1)
real :: ut(1,n+1,1),vt(1,n+1,1),tt(1,n+1,1),qt(1,n+1,1),qct(1,n+1,1),qit(1,n+1,1)
real :: kh(1,n+1,1),km(1,n+1,1)
real :: ps(1,1),zn(1,1),us(1,1),hp(1,1),pm(1,1),ph(1,1),xl(1,1),hf(1,1),qf(1,1)
real :: ws(1,1),br(1,1),u10(1,1),v10(1,1),uo(1,1),vo(1,1),topo(1,1),topo2(1,1)
real :: wstar(1,1),delta(1,1)
real, parameter :: r_d=287., r_v=461.6, cp=7.*r_d/2., g=9.81
real, parameter :: rcp=r_d/cp, rovg=r_d/g, ep_1=r_v/r_d-1., ep_2=r_d/r_v
real, parameter :: karman=0.4, xlv=2.5e6
integer :: kp(1,1), errflg
character(len=256) :: errmsg
logical :: flag_qc, flag_qi
u(1,:,1)=profiles(:,1);v(1,:,1)=profiles(:,2);t(1,:,1)=profiles(:,3)
q(1,:,1)=profiles(:,4);qc(1,:,1)=profiles(:,5);qi(1,:,1)=profiles(:,6)
p(1,:,1)=profiles(:,7);pf(1,:,1)=profiles(:,8);pi(1,:,1)=profiles(:,9)
dz(1,:,1)=profiles(:,10);rthr(1,:,1)=profiles(:,11)
ut=0.;vt=0.;tt=0.;qt=0.;qct=0.;qit=0.;kh=0.;km=0.
ps=surface(1);zn=surface(2);us=surface(3);pm=surface(4);ph=surface(5)
xl=surface(6);hf=surface(7);qf=surface(8);ws=surface(9);br=surface(10)
u10=surface(11);v10=surface(12);uo=surface(13);vo=surface(14)
topo=surface(15);topo2=surface(16)
hp=0.;wstar=0.;delta=0.;kp=0
flag_qc = fqc /= 0
flag_qi = fqi /= 0
call ysu(u3d=u,v3d=v,t3d=t,qv3d=q,qc3d=qc,qi3d=qi,p3d=p,p3di=pf,pi3d=pi, &
 rublten=ut,rvblten=vt,rthblten=tt,rqvblten=qt,rqcblten=qct,rqiblten=qit, &
 flag_qc=flag_qc,flag_qi=flag_qi, &
 cp=cp,g=g,rovcp=rcp,rd=r_d,rovg=rovg,ep1=ep_1,ep2=ep_2,karman=karman,xlv=xlv,rv=r_v, &
 dz8w=dz,psfc=ps,znt=zn,ust=us,hpbl=hp,psim=pm,psih=ph,xland=xl,hfx=hf,qfx=qf,wspd=ws,br=br, &
 dt=surface(17),kpbl2d=kp,exch_h=kh,exch_m=km,wstar=wstar,delta=delta,u10=u10,v10=v10, &
 uoce=uo,voce=vo,rthraten=rthr,ysu_topdown_pblmix=topdown,ctopo=topo,ctopo2=topo2, &
 idiff=0,flag_bep=.false., &
 ids=1,ide=2,jds=1,jde=2,kds=1,kde=n+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=n+1, &
 its=1,ite=1,jts=1,jte=1,kts=1,kte=n,errmsg=errmsg,errflg=errflg)
result=0.
result(:,1)=ut(1,:,1);result(:,2)=vt(1,:,1);result(:,3)=tt(1,:,1);result(:,4)=qt(1,:,1)
result(:,5)=qct(1,:,1);result(:,6)=qit(1,:,1);result(:,7)=kh(1,:,1);result(:,8)=km(1,:,1)
result(1,9)=hp(1,1);result(2,9)=real(kp(1,1));result(3,9)=wstar(1,1);result(4,9)=delta(1,1)
result(5,9)=u10(1,1);result(6,9)=v10(1,1)
end subroutine
