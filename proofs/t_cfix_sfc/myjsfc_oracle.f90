! Public-call wrapper around the UNMODIFIED WRF 4.7.1 module_sf_myjsfc.F (Janjic Eta, sf_sfclay_physics=2).
! Mirrors module_surface_driver.F:2210-2224 (CALL MYJSFC(itimestep,ht,dz8w,p_phy,p8w,th_phy,t_phy,qv_curr,qc_curr,
! u_phy,v_phy,tke_pbl,tsk,qsfc,thz0,qz0,uz0,vz0,lowlyr,xland,ivgtyp,isurban,iz0tlnd,ust,znt,z0,pblh,mavail,rmol,
! akhs,akms,br,chs,chs2,cqs2,hfx,qfx,lh,flhc,flqc,qgh,cpm,ct,u10,v10,t2,th2,tshltr,th10,q2,qshltr,q10,pshltr,
! p1000mb,u10e,v10e,...)) with LOWLYR=1 (MYJSFCINIT ARW), iz0tlnd=0, isurban=1. Lane t-cfix-sfc.
! prof(n,nz+1,10): dz8w pmid pint th t qv qc u v q2(TKE)   (pint uses nz+1 rows, others nz)
! sfc(n,14): ht tsk qsfc thz0 qz0 uz0 vz0 xland ust znt z0base mavail akhs akms
! out(n,36): ust znt pblh rmol akhs akms rib chs chs2 cqs2 hfx qfx lh flhc flqc qgh cpm ct u10 v10
!            t02 th02 tshltr th10 q02 qshltr q10 pshltr u10e v10e qsfc thz0 qz0 uz0 vz0 (36 unused)
subroutine probe(n, nz, itimestep, prof, sfc, out) bind(C)
use iso_c_binding
use module_sf_myjsfc
implicit none
integer(c_int), value :: n, nz, itimestep
real(c_float), intent(in) :: prof(n,nz+1,10), sfc(n,14)
real(c_float), intent(out) :: out(n,36)
real, parameter :: p1000=100000.
real, dimension(n,nz+1,1) :: dz,pmid,pint,th,t,qv,qc,u,v,q2
real, dimension(n,1) :: ht,tsk,qsfc,thz0,qz0,uz0,vz0,xland,ust,znt,z0b,mavail,akhs,akms
real, dimension(n,1) :: pblh,rmol,rib,chs,chs2,cqs2,hfx,qfx,lh,flhc,flqc,qgh,cpm,ct,u10,v10
real, dimension(n,1) :: t02,th02,tshltr,th10,q02,qshltr,q10,pshltr,u10e,v10e
integer :: lowlyr(n,1), ivgtyp(n,1)
real :: xice0(n,1)
logical, save :: init=.false.
dz(:,:,1)=prof(:,:,1); pmid(:,:,1)=prof(:,:,2); pint(:,:,1)=prof(:,:,3); th(:,:,1)=prof(:,:,4)
t(:,:,1)=prof(:,:,5); qv(:,:,1)=prof(:,:,6); qc(:,:,1)=prof(:,:,7); u(:,:,1)=prof(:,:,8)
v(:,:,1)=prof(:,:,9); q2(:,:,1)=prof(:,:,10)
ht(:,1)=sfc(:,1); tsk(:,1)=sfc(:,2); qsfc(:,1)=sfc(:,3); thz0(:,1)=sfc(:,4); qz0(:,1)=sfc(:,5)
uz0(:,1)=sfc(:,6); vz0(:,1)=sfc(:,7); xland(:,1)=sfc(:,8); ust(:,1)=sfc(:,9); znt(:,1)=sfc(:,10)
z0b(:,1)=sfc(:,11); mavail(:,1)=sfc(:,12); akhs(:,1)=sfc(:,13); akms(:,1)=sfc(:,14)
lowlyr=1; ivgtyp=2; pblh=0.; rmol=0.; xice0=0.
if (.not. init) then
  ! physics_init MYJSFCINIT: FH01/FH02, ZTMIN/ZTMAX/DZETA and the PSIM/PSIH tables. RESTART=.true.
  ! keeps the caller's USTAR/Z0 operands (the cold USTAR=0.1 is reproduced by MYJSFC's NTSD==1 branch).
  call myjsfcinit(lowlyr,ust,z0b,xland,xice0,ivgtyp,.true.,.true., &
       1,n+1,1,2,1,nz+1, 1,n,1,1,1,nz+1, 1,n,1,1,1,nz)
  init=.true.
end if
call myjsfc(itimestep,ht,dz,pmid,pint,th,t,qv,qc,u,v,q2, &
     tsk,qsfc,thz0,qz0,uz0,vz0,lowlyr,xland,ivgtyp,1,0, &
     ust,znt,z0b,pblh,mavail,rmol,akhs,akms,rib, &
     chs,chs2,cqs2,hfx,qfx,lh,flhc,flqc,qgh,cpm,ct, &
     u10,v10,t02,th02,tshltr,th10,q02,qshltr,q10,pshltr,p1000,u10e,v10e, &
     1,n+1,1,2,1,nz+1, 1,n,1,1,1,nz+1, 1,n,1,1,1,nz)
out=0.
out(:,1)=ust(:,1); out(:,2)=znt(:,1); out(:,3)=pblh(:,1); out(:,4)=rmol(:,1); out(:,5)=akhs(:,1)
out(:,6)=akms(:,1); out(:,7)=rib(:,1); out(:,8)=chs(:,1); out(:,9)=chs2(:,1); out(:,10)=cqs2(:,1)
out(:,11)=hfx(:,1); out(:,12)=qfx(:,1); out(:,13)=lh(:,1); out(:,14)=flhc(:,1); out(:,15)=flqc(:,1)
out(:,16)=qgh(:,1); out(:,17)=cpm(:,1); out(:,18)=ct(:,1); out(:,19)=u10(:,1); out(:,20)=v10(:,1)
out(:,21)=t02(:,1); out(:,22)=th02(:,1); out(:,23)=tshltr(:,1); out(:,24)=th10(:,1); out(:,25)=q02(:,1)
out(:,26)=qshltr(:,1); out(:,27)=q10(:,1); out(:,28)=pshltr(:,1); out(:,29)=u10e(:,1); out(:,30)=v10e(:,1)
out(:,31)=qsfc(:,1); out(:,32)=thz0(:,1); out(:,33)=qz0(:,1); out(:,34)=uz0(:,1); out(:,35)=vz0(:,1)
end subroutine probe
