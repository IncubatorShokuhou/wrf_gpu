! Full unmodified module_fdda_spnudging.F + FFTPACK5 oracle.
! Wrapper uses the fddagd_driver's single tile mass footprint and kte=nz;
! patch extents include ide/jde/kde, as first_rk_step_part1 passes them.
! All binary 3-D arrays: rectangular Fortran(i,k,j), zero-filled halos.
program spectral_oracle
  use module_fdda_spnudging
  use module_domain, only: domain
  implicit none
  type(domain) :: grid
  integer :: nx,ny,nz,ncall,c,n,i,j,k,stepfg,fgdtzero
  integer :: interval, end_h, ip(4), iz(4), kz(4), dk(4), ktrop,xwave,ywave,iramp
  integer :: istep,secs,days,seconds
  real :: guv,gt,gph,gq,ramp,dt,xtime,minutes
  real, allocatable :: model(:,:,:,:),old(:,:,:,:),new(:,:,:,:),rate(:,:,:,:)
  real, allocatable :: pblh(:,:),ht(:,:),zw(:,:,:),z(:,:,:)
  real, allocatable :: c1h(:),c2h(:),c1f(:),c2f(:),mu2(:,:),mub(:,:),muts(:,:),mut(:,:),muu(:,:),muv(:,:)
  real, allocatable :: owned(:,:,:,:)
  integer :: ib(1),ie(1),jb(1),je(1)
  character(len=512) :: inpath,outpath
  call get_command_argument(1,inpath)
  call get_command_argument(2,outpath)
  open(10,file=trim(inpath),access='stream',form='unformatted',status='old')
  read(10) nx,ny,nz,ncall
  read(10) interval,end_h,ip,iz,kz,dk,ktrop,xwave,ywave,iramp,stepfg,fgdtzero
  read(10) guv,gt,gph,gq,ramp,dt
  allocate(model(nx+1,nz+1,ny+1,5),old(nx+1,nz+1,ny+1,5),new(nx+1,nz+1,ny+1,5))
  allocate(rate(nx+1,nz+1,ny+1,5),pblh(nx+1,ny+1),ht(nx+1,ny+1),zw(nx+1,nz+1,ny+1),z(nx+1,nz+1,ny+1))
  allocate(grid%dif_analysis(nx+1,nz+1,ny+1))
  allocate(c1h(nz),c2h(nz),c1f(nz+1),c2f(nz+1),mu2(nx,ny),mub(nx,ny),muts(nx,ny),mut(nx,ny),muu(nx,ny),muv(nx,ny))
  allocate(owned(nx+1,nz+1,ny+1,5))
  read(10) c1h,c2h,c1f,c2f
  rate=0.
  ib=1; ie=nx; jb=1; je=ny
  open(11,file=trim(outpath),access='stream',form='unformatted',status='replace')
  do c=1,ncall
    read(10) istep,secs
    read(10) model,old,new,pblh,ht,zw
    read(10) mu2,mub,muts
    days=secs/86400
    seconds=secs-days*86400
    xtime=real(days)*24.*60.+real(seconds)/60.
    minutes=real(dble(secs)/60.d0)
    z=0.;grid%dif_analysis=0.
    ! fddagd_driver: no OLD overwrite for SPNUDGING; held rates reset by fgdtzero.
    if (mod(istep-1,stepfg)==0 .and. fgdtzero==1) rate=0.
    if (istep==1 .or. mod(istep,stepfg)==0) then
    rate=0.
    call spectral_nudging(grid,istep,dt,xtime,minutes,1,interval,end_h, &
      ip(1),ip(2),ip(3),ip(4), &
      iz(1),kz(1),dk(1),iz(2),kz(2),dk(2),iz(3),kz(3),dk(3),iz(4),kz(4),dk(4),ktrop, &
      guv,gt,gph,gq,iramp,ramp,xwave,ywave, &
      model(:,:,:,1),model(:,:,:,2),model(:,:,:,3),model(:,:,:,4),model(:,:,:,5), &
      old(:,:,:,1),old(:,:,:,2),old(:,:,:,3),old(:,:,:,4),old(:,:,:,5), &
      new(:,:,:,1),new(:,:,:,2),new(:,:,:,3),new(:,:,:,4),new(:,:,:,5), &
      rate(:,:,:,1),rate(:,:,:,2),rate(:,:,:,3),rate(:,:,:,4),rate(:,:,:,5), &
      pblh,ht,z,zw, &
      1,nx+1,1,ny+1,1,nz+1, &
      1,nx+1,1,ny+1,1,nz+1, &
      ib,ie,jb,je,1,nz,1, &
      1,nx+1,1,ny+1,1,nz+1, &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1, &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    endif
    write(11) rate
    ! Literal calculate_full + calc_mu_uv + calculate_phy_tend FDDA block.
    ! PH is deliberately NOT coupled here: module_em.F accepts no RPHNDGDTEN.
    mut=mub+mu2
    do j=1,ny
    do i=2,nx
      muu(i,j)=0.5*(mu2(i,j)+mu2(i-1,j)+mub(i,j)+mub(i-1,j))
    enddo
    enddo
    do j=2,ny
    do i=1,nx
      muv(i,j)=0.5*(mu2(i,j)+mu2(i,j-1)+mub(i,j)+mub(i,j-1))
    enddo
    enddo
    do j=1,ny
    do k=1,nz
    do i=2,nx
      rate(i,k,j,1)=(c1h(k)*muu(i,j)+c2h(k))*rate(i,k,j,1)
    enddo
    enddo
    enddo
    do j=2,ny
    do k=1,nz
    do i=1,nx
      rate(i,k,j,2)=(c1h(k)*muv(i,j)+c2h(k))*rate(i,k,j,2)
    enddo
    enddo
    enddo
    do j=1,ny
    do k=1,nz
    do i=1,nx
      rate(i,k,j,3)=(c1h(k)*mut(i,j)+c2h(k))*rate(i,k,j,3)
      rate(i,k,j,5)=(c1h(k)*mut(i,j)+c2h(k))*rate(i,k,j,5)
    enddo
    enddo
    enddo
    write(11) rate
    ! phy_fg_ten specified ownership: add_c2c_u, add_c2c_v, add_a2a, add_a2a_ph.
    owned=0.
    owned(2:nx,1:nz,2:ny-1,1)=rate(2:nx,1:nz,2:ny-1,1)
    owned(2:nx-1,1:nz,2:ny,2)=rate(2:nx-1,1:nz,2:ny,2)
    owned(2:nx-1,1:nz,2:ny-1,3)=rate(2:nx-1,1:nz,2:ny-1,3)
    owned(2:nx-1,1:nz+1,2:ny-1,4)=rate(2:nx-1,1:nz+1,2:ny-1,4)
    owned(2:nx-1,1:nz,2:ny-1,5)=rate(2:nx-1,1:nz,2:ny-1,5)
    write(11) owned
    ! calc_mu_uv_1(muts), phy_prep_part2. SPNUDGING decouples PH, NOT Q.
    do j=1,ny
    do i=2,nx
      muu(i,j)=0.5*(muts(i,j)+muts(i-1,j))
    enddo
    enddo
    do j=2,ny
    do i=1,nx
      muv(i,j)=0.5*(muts(i,j)+muts(i,j-1))
    enddo
    enddo
    do j=1,ny
    do k=1,nz
    do i=2,nx
      rate(i,k,j,1)=rate(i,k,j,1)/(c1h(k)*muu(i,j)+c2h(k))
    enddo
    enddo
    enddo
    do j=2,ny
    do k=1,nz
    do i=1,nx
      rate(i,k,j,2)=rate(i,k,j,2)/(c1h(k)*muv(i,j)+c2h(k))
    enddo
    enddo
    enddo
    do j=1,ny
    do k=1,nz
    do i=1,nx
      rate(i,k,j,3)=rate(i,k,j,3)/(c1h(k)*muts(i,j)+c2h(k))
    enddo
    enddo
    enddo
    do j=1,ny
    do k=1,nz+1
    do i=1,nx
      rate(i,k,j,4)=rate(i,k,j,4)/(c1f(k)*muts(i,j)+c2f(k))
    enddo
    enddo
    enddo
    write(11) rate
  enddo
  close(10);close(11)
end program
