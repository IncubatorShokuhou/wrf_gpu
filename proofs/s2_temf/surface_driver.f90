! Pristine surface_driver caller mapping, WRF REAL and WRF constants.
! The full 3-D temfsfclay entry point checks its own first-level view.
program surface_oracle
  use module_sf_temfsfclay
  use module_model_constants, only: cp,g,rcp,r_d,xlv,svp1,svp2,svp3,svpt0,ep_1,ep_2,karman
  implicit none
  integer :: n,i,s,k
  integer, parameter :: nz=4
  real, allocatable :: a(:,:,:),u(:,:,:),v(:,:,:),th(:,:,:),qv(:,:,:),p(:,:,:),pi(:,:,:),rho(:,:,:),z(:,:,:),te(:,:,:)
  real, allocatable :: ht(:,:),psfc(:,:),znt(:,:),mavail(:,:),xland(:,:),tsk(:,:),f(:,:),hfx(:,:),hd(:,:),wm(:,:)
  real, allocatable :: chs(:,:),chs2(:,:),cqs2(:,:),cpm(:,:),ust(:,:),qfx(:,:),lh(:,:),flhc(:,:),flqc(:,:)
  real, allocatable :: qgh(:,:),qsfc(:,:),u10(:,:),v10(:,:),th2(:,:),t2(:,:),q2(:,:),exch(:,:)
  real :: vals(18)
  read(*,*) n
  allocate(a(n,nz,1),u(n,nz,1),v(n,nz,1),th(n,nz,1),qv(n,nz,1),p(n,nz,1),pi(n,nz,1),rho(n,nz,1),z(n,nz,1),te(n,nz,1))
  allocate(ht(n,1),psfc(n,1),znt(n,1),mavail(n,1),xland(n,1),tsk(n,1),f(n,1),hfx(n,1),hd(n,1),wm(n,1))
  allocate(chs(n,1),chs2(n,1),cqs2(n,1),cpm(n,1),ust(n,1),qfx(n,1),lh(n,1),flhc(n,1),flqc(n,1))
  allocate(qgh(n,1),qsfc(n,1),u10(n,1),v10(n,1),th2(n,1),t2(n,1),q2(n,1),exch(n,1))
  a=0.; te=17.; xland=1.; chs=0.; chs2=0.; cqs2=0.; cpm=0.; ust=0.; qfx=0.; lh=0.; flhc=0.; flqc=0.; qgh=0.; qsfc=0.
  do i=1,n
    read(*,*) vals
    do k=1,nz
      u(i,k,1)=vals(1); v(i,k,1)=vals(2); th(i,k,1)=vals(3); qv(i,k,1)=vals(4)
      p(i,k,1)=vals(5); pi(i,k,1)=vals(6); rho(i,k,1)=vals(7); z(i,k,1)=vals(8)+100.*(k-1)
    end do
    ht(i,1)=vals(9); psfc(i,1)=vals(10); znt(i,1)=vals(11); mavail(i,1)=vals(12)
    tsk(i,1)=vals(13); f(i,1)=vals(14); hfx(i,1)=vals(15); te(i,1,1)=vals(16)
    hd(i,1)=vals(17); wm(i,1)=vals(18)
  end do
  do s=1,3
    call temfsfclay(u,v,th,qv,p,pi,rho,z,ht,cp,g,rcp,r_d,xlv,psfc,chs,chs2,cqs2,cpm, &
         znt,ust,mavail,xland,hfx,qfx,lh,tsk,flhc,flqc,qgh,qsfc,u10,v10,th2,t2,q2, &
         svp1,svp2,svp3,svpt0,ep_1,ep_2,karman,f,te,hd,exch,wm, &
         1,n+1,1,2,1,nz, 1,n,1,1,1,nz, 1,n,1,1,1,nz)
    do i=1,n
      write(*,'(2(I5,1X),20(ES24.16E3,1X))') s,i,te(i,1,1),wm(i,1),ust(i,1), &
        flhc(i,1),flqc(i,1),exch(i,1),chs(i,1),hfx(i,1),qfx(i,1),lh(i,1),qsfc(i,1), &
        u10(i,1),v10(i,1),t2(i,1),th2(i,1),q2(i,1),chs2(i,1),cqs2(i,1),qgh(i,1),cpm(i,1)
    end do
    if (any(te(:,2:nz,1)/=17.)) stop 7
  end do
end program

subroutine wrf_debug(level,message)
  integer :: level
  character(*) :: message
end subroutine
