! Single-slab boundary of WRF fddaobs_driver -> nudob, root domain, nscan=0.
! Input/output: native-endian stream; slab arrays in Fortran (i,k) order.
program driver
  use obs_oracle
  implicit none
  integer :: nx,nz,ny,nobs,ncalls,ivar,j,ionf,enabled,nudge,horiz,vert
  integer :: c,n,i,k,ktau,ktaur,rest,levidn(1),parid(1)
  real :: dx,rinxy,twindo,dtmin,coef,sfcfact,sfcfacr,dpsmx,xtime,tfaci
  real, allocatable :: rio(:),rjo(:),rko(:),timeob(:),varobs(:,:),errf(:,:)
  real, allocatable :: lev(:),plfo(:),nlevs(:),mu(:),msfx(:),msfy(:),c1h(:),c2h(:)
  real, allocatable :: pbase(:,:),pp(:,:),z(:,:),qvb(:,:,:),terrain(:),vih1(:),vih2(:)
  real, allocatable :: aten(:,:),savwt(:,:,:),obpbl(:)
  integer, allocatable :: kpbl(:)
  character(1024) :: inpath,outpath
  call get_command_argument(1,inpath)
  call get_command_argument(2,outpath)
  open(10,file=trim(inpath),access='stream',form='unformatted',status='old')
  read(10) nx,nz,ny,nobs,ncalls,ivar,j,ionf,enabled,nudge,horiz,vert
  read(10) dx,rinxy,twindo,dtmin,coef,sfcfact,sfcfacr,dpsmx
  allocate(rio(nobs),rjo(nobs),rko(nobs),timeob(nobs),varobs(6,nobs),errf(10,nobs),obpbl(nobs))
  allocate(lev(nobs),plfo(nobs),nlevs(nobs),mu(nx),msfx(nx),msfy(nx),c1h(nz),c2h(nz))
  allocate(pbase(nx,nz),pp(nx,nz),z(nx,nz),qvb(nx,nz,ny),terrain(nx),vih1(nx),vih2(nx))
  allocate(aten(nx,nz),savwt(6,nx,nz),kpbl(nx))
  read(10) rio,rjo,rko,timeob,varobs,errf
  read(10) mu,c1h,c2h,msfx,msfy,pbase,pp,kpbl,vih1,vih2,terrain,z
  read(10) aten,savwt
  lev=1.; nlevs=1.; plfo=1.; levidn=0; parid=0; qvb=0.
  open(11,file=trim(outpath),access='stream',form='unformatted',status='replace')
  do c=1,ncalls
    read(10) ktau,ktaur,rest
    read(10) xtime,tfaci
    ! fddaobs_driver passes: temp/mois/wind flags=1, nscan=0, inest=1;
    ! only this IVAR coefficient is read. j bound guard is also in nudob.
    call nudob(j,ivar,aten,1,rest==1,ktau,ktaur,xtime,mu,c1h,c2h,msfx,msfy, &
        6,10,nobs,1,0,ionf,rinxy,twindo,nudge==1,sfcfact,sfcfacr,levidn,parid,nobs, &
        1.,2.,50.,1./dpsmx,tfaci,horiz,vert,20.,lev,plfo,nlevs, &
        1,dx,dtmin,rio,rjo,rko,timeob,varobs,errf,pbase,5000.,pp, &
        enabled,enabled,enabled,coef,coef,coef,savwt,kpbl,0,vih1,vih2,terrain,z,.false., &
        1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,qvb,0)
    write(11) aten,savwt(ivar,:,:)
  enddo
  close(10); close(11)
end program driver
