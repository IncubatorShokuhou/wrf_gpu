! Public-call wrapper around the UNMODIFIED WRF 4.7.1 module_bl_gfs.F (BL_GFS -> MONINP, bl_pbl_physics=3).
! Mirrors module_pbl_driver.F:1404-1424: CP=cp, G=g, ROVCP=rcp, R=r_d, ROVG=rovg, EP1=ep_1, KARMAN=karman
! (share/module_model_constants.F), DT=dtbl, P_FIRST_SCALAR=2, P_QI=3 (qi present) or 1. Lane t-cfix-sfc.
! prof(n,nz+1,11): u v th t qv qc qi p pi dz8w z   (row nz+1 copies nz; WRF kme=kte+1)
! sfc(n,11): psfc ust pbl psim psih hfx qfx tsk(TSKOLD) gz1oz0 wspd br
! tend(n,nz,6): rublten rvblten rthblten rqvblten rqcblten rqiblten
! diag(n,4): ust pbl wspd kpbl
subroutine probe(n, nz, has_qi, dt, prof, sfc, tend, diag) bind(C)
use iso_c_binding
use module_model_constants, only: cp, g, rcp, r_d, rovg, ep_1, karman
use module_bl_gfs
implicit none
integer(c_int), value :: n, nz, has_qi
real(c_float), value :: dt
real(c_float), intent(in) :: prof(n,nz+1,11), sfc(n,11)
real(c_float), intent(out) :: tend(n,nz,6), diag(n,4)
real, dimension(n,nz+1,1) :: u,v,th,t,qv,qc,qi,p,pii,dz,z,ru,rv,rth,rqv,rqc,rqi
real, dimension(n,1) :: psfc,ust,pbl,psim,psih,hfx,qfx,tsk,gz1oz0,wspd,br
integer :: kpbl(n,1), p_qi
u(:,:,1)=prof(:,:,1); v(:,:,1)=prof(:,:,2); th(:,:,1)=prof(:,:,3); t(:,:,1)=prof(:,:,4)
qv(:,:,1)=prof(:,:,5); qc(:,:,1)=prof(:,:,6); qi(:,:,1)=prof(:,:,7); p(:,:,1)=prof(:,:,8)
pii(:,:,1)=prof(:,:,9); dz(:,:,1)=prof(:,:,10); z(:,:,1)=prof(:,:,11)
psfc(:,1)=sfc(:,1); ust(:,1)=sfc(:,2); pbl(:,1)=sfc(:,3); psim(:,1)=sfc(:,4); psih(:,1)=sfc(:,5)
hfx(:,1)=sfc(:,6); qfx(:,1)=sfc(:,7); tsk(:,1)=sfc(:,8); gz1oz0(:,1)=sfc(:,9); wspd(:,1)=sfc(:,10); br(:,1)=sfc(:,11)
ru=0.; rv=0.; rth=0.; rqv=0.; rqc=0.; rqi=0.; kpbl=0
p_qi=1
if (has_qi /= 0) p_qi=3
call bl_gfs(u,v,th,t,qv,qc,qi,p,pii,ru,rv,rth,rqv,rqc,rqi, &
     cp,g,rcp,r_d,rovg,p_qi,2,dz,z,psfc,ust,pbl,psim,psih, &
     hfx,qfx,tsk,gz1oz0,wspd,br,dt,kpbl,ep_1,karman, &
     1,n+1,1,2,1,nz+1, 1,n,1,1,1,nz+1, 1,n,1,1,1,nz)
tend(:,:,1)=ru(:,1:nz,1); tend(:,:,2)=rv(:,1:nz,1); tend(:,:,3)=rth(:,1:nz,1)
tend(:,:,4)=rqv(:,1:nz,1); tend(:,:,5)=rqc(:,1:nz,1); tend(:,:,6)=rqi(:,1:nz,1)
diag(:,1)=ust(:,1); diag(:,2)=pbl(:,1); diag(:,3)=wspd(:,1); diag(:,4)=real(kpbl(:,1))
end subroutine probe
