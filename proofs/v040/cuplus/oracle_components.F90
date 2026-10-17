! Component oracle: all numerical procedures are copied byte-for-byte from
! pristine WRF by generate.py. This driver defines only their caller inputs.
program oracle_components
  use kfcup_oracle_components
  implicit none
  integer :: mode,n,ishall,kbottom,ktop
  real :: sfc,ez,ss,se,thbin,rbin,wstar,wparcel,pblh,active,dt
  real :: taucloud,tactive,tstar,lnint
  real, allocatable :: theta(:),moisture(:),jfd(:,:)
  real, allocatable :: ql(:),qi(:),qv(:),t(:),z(:),p(:),mf(:),cf(:),cft(:),lnterms(:)
  real, allocatable :: cf3(:,:,:),qc3(:,:,:),cup3(:,:,:)
  integer, allocatable :: flag3(:,:,:)
  real :: shall2(1,1),bot2(1,1),top2(1,1)
  integer :: icloud,forced
  read(*,*) mode,n
  allocate(ql(n),qi(n),qv(n),t(n),z(n),p(n),mf(n),cf(n),cft(n),lnterms(n))
  select case(mode)
  case(1)
    allocate(theta(n),moisture(n),jfd(n,n))
    read(*,*) sfc,ez,ss,se,thbin,rbin
    call cup_jfd(sfc,ez,ss,se,n,thbin,rbin,theta,moisture,jfd)
    write(*,'(*(ES25.16E3,1X))') sfc,ez,ss,se,theta,moisture,jfd
  case(2)
    read(*,*) ishall,kbottom,ktop
    read(*,*) wstar,wparcel,pblh,active,dt
    read(*,*) ql
    read(*,*) qi
    read(*,*) qv
    read(*,*) t
    read(*,*) z
    read(*,*) p
    read(*,*) mf
    call cupCloudFraction(ql,qi,qv,t,z,p,kbottom,ktop,ishall,wstar,wparcel,pblh,dt,active, &
         cf,cft,taucloud,tactive,tstar,lnterms,lnint,1,n,mf)
    ! Only the outputs actually assigned in the selected original branch.
    write(*,'(*(ES25.16E3,1X))') cf
    if (ishall /= 2) write(*,'(*(ES25.16E3,1X))') tstar
    if (ishall == 1) write(*,'(*(ES25.16E3,1X))') taucloud,tactive,lnint,lnterms
  case(3)
    read(*,*) t
    read(*,*) p
    read(*,*) qv
    do kbottom=1,n
      write(*,'(*(ES25.16E3,1X))') findCp(qv(kbottom)),findRs(t(kbottom),p(kbottom)),findRsi(t(kbottom),p(kbottom))
    enddo
  case(4)
    allocate(cf3(1,n,1),qc3(1,n,1),cup3(1,n,1),flag3(1,n,1))
    read(*,*) shall2,bot2,top2,icloud,forced
    read(*,*) cf3
    read(*,*) qc3
    read(*,*) cup3
    read(*,*) flag3
    call cup_radiation(1,1,1,1,1,n,10,icloud,forced/=0,shall2,bot2,top2,flag3,cf3,qc3,cup3)
    write(*,'(*(ES25.16E3,1X))') cf3,qc3,cup3
  case default
    stop 2
  end select
end program oracle_components
