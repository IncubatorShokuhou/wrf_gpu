! RAD_FLG and ETARA routine-boundary oracle. Input arrays are supplied explicitly;
! interface temperature and eta are fixture operands, not reconstructed truth.
program radiation_oracle
  use module_ra_flg, only: rad_flg
  use module_ra_gfdleta, only: etara, gfdletainit, cal_mon_day
  use module_model_constants, only: g, cp, degrad
  use module_configure, only: grid_config_rec_type
  implicit none
  integer :: nc,nz,i,k,u,scheme,yr,jmonth,jday
  character(len=1024) :: input,output,arg
  type(grid_config_rec_type) :: config
  real :: ptop,gmt,radt,dt
  real, allocatable :: znw(:),znu(:),julian(:),tau(:),declin(:),solcon(:)
  integer, allocatable :: julday(:)
  real, allocatable :: sfc(:,:),fields(:,:,:),tout(:,:,:),fout(:,:)
  real, allocatable :: t(:,:,:),p(:,:,:),p8w(:,:,:),pi(:,:,:),rho(:,:,:),dz(:,:,:),t8w(:,:,:)
  real, allocatable :: qv(:,:,:),qc(:,:,:),qr(:,:,:),qi(:,:,:),qs(:,:,:),qg(:,:,:),cf(:,:,:),o3(:,:,:)
  real, allocatable :: th(:,:,:),lw(:,:,:),sw(:,:,:)
  real :: alb(1,1),emiss(1,1),tsk(1,1),lat(1,1),lon(1,1),xland(1,1),snow(1,1),veg(1,1)
  real :: glw(1,1),gsw(1,1),swdn(1,1),swdnc(1,1),swtoa(1,1),lwtoa(1,1),cz(1,1)
  real :: dir(1,1),dif(1,1),dni(1,1),htop(1,1),hbot(1,1),htopr(1,1),hbotr(1,1),cuppt(1,1)
  real :: cfl(1,1),cfm(1,1),cfh(1,1),acst(1,1),accv(1,1)
  integer :: ncst(1,1),nccv(1,1)
  call get_command_argument(1,arg); read(arg,*) scheme
  call get_command_argument(2,input); call get_command_argument(3,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u) nc,nz,ptop,gmt,radt,dt,yr
  allocate(znw(nz+1),znu(nz+1),julian(nc),tau(nc),declin(nc),solcon(nc),julday(nc))
  allocate(sfc(nc,11),fields(nc,nz+1,15),tout(nc,nz,3),fout(nc,14))
  read(u) znw,znu,julian,tau,declin,solcon,julday,sfc,fields
  close(u)
#ifdef FLG_SAVEPOINTS
  if (scheme == 7) open(unit=93,file=trim(output)//'.capture',access='stream',form='unformatted',status='replace')
#endif
#ifdef GFDL_SAVEPOINTS
  if (scheme == 99) open(unit=95,file=trim(output)//'.capture',access='stream',form='unformatted',status='replace')
#endif
#ifdef GFDL_SW_SAVEPOINTS
  if (scheme == 99) open(unit=97,file=trim(output)//'.swr93',access='stream',form='unformatted',status='replace')
#endif
#ifdef GFDL_LW_SAVEPOINTS
  if (scheme == 99) open(unit=98,file=trim(output)//'.lwr88',access='stream',form='unformatted',status='replace')
#endif
#ifdef GFDL_RADFS_SAVEPOINTS
  if (scheme == 99) open(unit=99,file=trim(output)//'.radfs',access='stream',form='unformatted',status='replace')
#endif
#ifdef GFDL_RADTN_SAVEPOINTS
  if (scheme == 99) open(unit=100,file=trim(output)//'.radtn',access='stream',form='unformatted',status='replace')
#endif
  allocate(t(1,nz+1,1),p(1,nz+1,1),p8w(1,nz+1,1),pi(1,nz+1,1),rho(1,nz+1,1))
  allocate(dz(1,nz+1,1),t8w(1,nz+1,1),qv(1,nz+1,1),qc(1,nz+1,1),qr(1,nz+1,1))
  allocate(qi(1,nz+1,1),qs(1,nz+1,1),qg(1,nz+1,1),cf(1,nz+1,1),o3(1,nz+1,1))
  allocate(th(1,nz+1,1),lw(1,nz+1,1),sw(1,nz+1,1))
  if (scheme == 99) then
    call cal_mon_day(julday(1),yr,jmonth,jday)
    call gfdletainit(emiss,znw,znu,ptop/1000.,yr,jmonth,jday,gmt,config,.true., &
      1,2,1,2,1,nz+1, 1,1,1,1,1,nz+1, 1,1,1,1,1,nz)
  endif
  do i=1,nc
    t(1,:,1)=fields(i,:,1); p(1,:,1)=fields(i,:,2); pi(1,:,1)=fields(i,:,3)
    rho(1,:,1)=fields(i,:,4); t8w(1,:,1)=fields(i,:,5); qv(1,:,1)=fields(i,:,6)
    qc(1,:,1)=fields(i,:,7); qr(1,:,1)=fields(i,:,8); qi(1,:,1)=fields(i,:,9)
    qs(1,:,1)=fields(i,:,10); qg(1,:,1)=fields(i,:,11); cf(1,:,1)=fields(i,:,12)
    p8w(1,:,1)=fields(i,:,13); dz(1,:,1)=fields(i,:,14); o3(1,:,1)=fields(i,:,15)
    tsk=sfc(i,1); emiss=sfc(i,2); alb=sfc(i,3); xland=sfc(i,4)
    lat=sfc(i,5); lon=sfc(i,6); snow=sfc(i,7); veg=sfc(i,8)
    htop=sfc(i,9); hbot=sfc(i,10); cuppt=sfc(i,11)
    th=0.; lw=0.; sw=0.; glw=0.; gsw=0.; swdn=0.; swdnc=0.; swtoa=0.; lwtoa=0.
    cz=0.; dir=0.; dif=0.; dni=0.; cfl=0.; cfm=0.; cfh=0.
    acst=0.; accv=0.; ncst=0; nccv=0; htopr=0.; hbotr=0.
    if (scheme == 7) then
      call rad_flg(peven=p8w,podd=p,t8w=t8w,degrees=t,pi3d=pi,o3=o3,g=g,cp=cp, &
        albedo=alb,tskin=tsk,h2o=qv,cld_iccld=qi,cld_wlcld=qc,cld_prwc=qr,cld_pgwc=qg,cld_snow=qs, &
        f_qv=.true.,f_qc=.true.,f_qr=.true.,f_qi=.true.,f_qs=.true.,f_qg=.true.,warm_rain=.false., &
        cloudstrf=cf,emiss=emiss,air_den=rho,dz3d=dz,solcon=solcon(i),declin=declin(i), &
        xtime=tau(i)*60.,xlong=lon,xlat=lat,julday=julday(i),gmt=gmt,radt=radt,degrad=degrad,dtcolumn=dt, &
        ids=1,ide=2,jds=1,jde=2,kds=1,kde=nz+1,ims=1,idim=1,jms=1,jdim=1,kms=1,kmax=nz+1, &
        its=1,ite=1,jts=1,jte=1,kts=1,kte=nz,uswtop=swtoa,ulwtop=lwtoa,netswbot=gsw, &
        dlwbot=glw,dswbot=swdn,deltat=th,dtshort=sw,dtlongwv=lw,swddir=dir,swddif=dif,swddni=dni)
    else if (scheme == 99) then
      ! GFDL init sets EMISS=1; ETARA itself uses unit surface emissivity.
      emiss=1.
      call gfdl_call(.true.,.false.)
      call gfdl_call(.true.,.true.)
    else
      stop 4
    endif
    tout(i,:,1)=th(1,1:nz,1); tout(i,:,2)=lw(1,1:nz,1); tout(i,:,3)=sw(1,1:nz,1)
    fout(i,:)=[glw(1,1),gsw(1,1),swdn(1,1),swdnc(1,1),swtoa(1,1),lwtoa(1,1), &
      dir(1,1),dif(1,1),dni(1,1),cfl(1,1),cfm(1,1),cfh(1,1),htopr(1,1),hbotr(1,1)]
  enddo
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
  write(u) nc,nz,tout,fout
  close(u)
#ifdef FLG_SAVEPOINTS
  if (scheme == 7) close(93)
#endif
#ifdef GFDL_SAVEPOINTS
  if (scheme == 99) close(95)
#endif
#ifdef GFDL_SW_SAVEPOINTS
  if (scheme == 99) close(97)
#endif
#ifdef GFDL_LW_SAVEPOINTS
  if (scheme == 99) close(98)
#endif
#ifdef GFDL_RADFS_SAVEPOINTS
  if (scheme == 99) close(99)
#endif
#ifdef GFDL_RADTN_SAVEPOINTS
  if (scheme == 99) close(100)
#endif
contains
  subroutine gfdl_call(do_lw,do_sw)
    logical, intent(in) :: do_lw,do_sw
    call etara(dt=dt,thraten=th,thratenlw=lw,thratensw=sw,cldfra=cf,pi3d=pi,xland=xland, &
      p8w=p8w,dz8w=dz,rho_phy=rho,p_phy=p,t=t,qv=qv,qw=qc,qi=qi,qs=qs, &
      tsk2d=tsk,glw=glw,rswin=swdn,gsw=gsw,rswinc=swdnc,rswtoa=swtoa,rlwtoa=lwtoa,czmean=cz, &
      glat=lat*degrad,glon=lon*degrad,htop=htop,hbot=hbot,htopr=htopr,hbotr=hbotr,albedo=alb,cuppt=cuppt, &
      vegfra=veg,snow=snow,g=g,gmt=gmt,nstepra=nint(radt*60./dt),nphs=1,itimestep=1, &
      xtime=tau(i)*60.,julian=julian(i),julyr=yr,julday=julday(i),gfdl_lw=do_lw,gfdl_sw=do_sw, &
      cfracl=cfl,cfracm=cfm,cfrach=cfh,acfrst=acst,ncfrst=ncst,acfrcv=accv,ncfrcv=nccv, &
      ids=1,ide=2,jds=1,jde=2,kds=1,kde=nz+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz+1, &
      its=1,ite=1,jts=1,jte=1,kts=1,kte=nz)
  end subroutine
end program
