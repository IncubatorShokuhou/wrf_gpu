! Staged ARW WRF QNSESFC -> QNSEPBL column, mfshconv=0.
! Constants and physics come from unmodified WRF sources; no port routines.
program oracle
  use module_model_constants, only: g, r_d, cp, ep_1
  use module_sf_qnsesfc, only: qnsesfc, qnsesfcinit
  use module_bl_qnsepbl, only: qnsepbl, qnsepblinit
  implicit none
  integer, parameter :: n=16
  real, dimension(1,n+1,1) :: u,v,t,th,qv,qc,p,pint,dz,exner,rho,q2,tke,exh,exm,el
  real, dimension(1,n+1,1) :: ru,rv,rth,rqv,rqc,wmf,lmf
  real, dimension(1,1) :: ht,tsk,xland,qsfc,thz0,qz0,uz0,vz0,ustar,znt,zbase
  real, dimension(1,1) :: pblh,rmol,akhs,akms,rib,chs,chs2,cqs2,hfx,qfx,lh,flhc,flqc,qgh,cpm,ct
  real, dimension(1,1) :: u10,v10,t02,th02,tshltr,th10,q02,qshltr,q10,pshltr,u10e,v10e
  real, dimension(1,1) :: mavail,sice,snow,chklowq,corf
  integer, dimension(1,1) :: lowlyr,ivg,kpbl
  integer :: cid,k,itimestep,stepbl,scm_flag
  real :: z,zm,theta0,lapse,heat,dt,ustseed
  character(32) :: arg

  call get_command_argument(1,arg)
  read(arg,*) cid
  if(cid<1.or.cid>15) stop 2
  theta0=295.; lapse=.004; heat=5.; xland=1.; ustseed=.4
  dt=30.; stepbl=1; ht=250.; corf=1.e-4; snow=0.; sice=0.; itimestep=2; scm_flag=0
  if(cid==1) lapse=-.002                   ! convective land
  if(cid==2) then                          ! stable night
    lapse=.018; heat=-6.; ustseed=.15
  endif
  if(cid>=3.and.cid<=5) then               ! water sublayer branches
    xland=2.; ustseed=.1
    if(cid==4) ustseed=.4
    if(cid==5) ustseed=1.
  endif
  if(cid==6) then                          ! snow latent heat conversion
    theta0=268.; heat=-2.; snow=12.
  endif
  if(cid==7) lapse=-.005                   ! cloudy convection
  if(cid==8) then                          ! weak shear, strong stratification
    lapse=.025; heat=-8.; ustseed=.08
  endif
  if(cid==9) corf=-1.e-4                   ! WRF signed Coriolis floor
  if(cid==10) itimestep=1                  ! first-step surface path
  if(cid==11) then                         ! no available surface moisture
    chklowq=0.; heat=0.
  endif
  if(cid==12) then                         ! inversion CT and additive tendencies
    lapse=-.002; stepbl=2
  endif
  if(cid==13) scm_flag=1                   ! forced surface flux preservation
  if(cid==14) scm_flag=-1                  ! absent optional preservation
  if(cid==15) then                         ! first-step water sublayer
    xland=2.; ustseed=.1; itimestep=1
  endif
  z=0.; pint(1,1,1)=97000.
  do k=1,n
    dz(1,k,1)=20.+15.*real(k-1)
    zm=z+.5*dz(1,k,1)
    th(1,k,1)=theta0+lapse*zm
    if(cid==12.and.k>=6) th(1,k,1)=theta0+.01*zm
    p(1,k,1)=97000.*exp(-g*zm/(r_d*theta0))
    exner(1,k,1)=(p(1,k,1)/1.e5)**(r_d/cp)
    t(1,k,1)=th(1,k,1)*exner(1,k,1)
    qv(1,k,1)=.012*exp(-zm/1800.)
    qc(1,k,1)=0.
    if(cid==7.and.k>=4.and.k<=8) qc(1,k,1)=.0005
    rho(1,k,1)=p(1,k,1)/(r_d*t(1,k,1)*(1.+ep_1*qv(1,k,1)))
    u(1,k,1)=3.+.006*zm; v(1,k,1)=1.-.002*zm
    if(cid==8) then
      u(1,k,1)=.01; v(1,k,1)=0.
    endif
    q2(1,k,1)=max(.005,.3*exp(-zm/600.))
    z=z+dz(1,k,1)
    pint(1,k+1,1)=97000.*exp(-g*z/(r_d*theta0))
  enddo
  tsk=theta0+heat; zbase=.08; znt=zbase; mavail=.6; ivg=10
  lowlyr=1; ustar=ustseed; pblh=500.; rmol=0.; akhs=.01; akms=.01
  thz0=tsk; qz0=.012; qsfc=.012; uz0=0.; vz0=0.
  hfx=heat*15.; qfx=1.e-5; lh=0.; ct=0.; chklowq=1.
  if(cid==11) chklowq=0.
  call qnsesfcinit(lowlyr,ustar,znt,xland,sice,ivg,.false.,.false., &
                  1,2,1,2,1,n+1, 1,1,1,1,1,n+1, 1,1,1,1,1,n)
  ! Restart/ordinary surface input is prescribed after table initialization.
  ustar=ustseed
  write(*,'(a)') 'INPUT_SFC'
  call dump_surface(.false.)
  if(scm_flag>=0) then
  call qnsesfc(itimestep,ht,dz,p,pint,th,t,qv,qc,u,v,q2, &
               tsk,qsfc,thz0,qz0,uz0,vz0,lowlyr,xland, &
               ustar,znt,zbase,pblh,mavail,rmol,akhs,akms,rib, &
               chs,chs2,cqs2,hfx,qfx,lh,flhc,flqc,qgh,cpm,ct, &
               u10,v10,t02,th02,tshltr,th10,q02,qshltr,q10,pshltr,u10e,v10e, &
               1,2,1,2,1,n+1, 1,1,1,1,1,n+1, 1,1,1,1,1,n,SCM_FORCE_FLUX=scm_flag)
  else
  call qnsesfc(itimestep,ht,dz,p,pint,th,t,qv,qc,u,v,q2, &
               tsk,qsfc,thz0,qz0,uz0,vz0,lowlyr,xland, &
               ustar,znt,zbase,pblh,mavail,rmol,akhs,akms,rib, &
               chs,chs2,cqs2,hfx,qfx,lh,flhc,flqc,qgh,cpm,ct, &
               u10,v10,t02,th02,tshltr,th10,q02,qshltr,q10,pshltr,u10e,v10e, &
               1,2,1,2,1,n+1, 1,1,1,1,1,n+1, 1,1,1,1,1,n)
  endif
  write(*,'(a)') 'OUTPUT_SFC'
  call dump_surface(.true.)
  call qnsepblinit(ru,rv,rth,rqv,tke,exh,.false.,.false., &
                  1,2,1,2,1,n+1, 1,1,1,1,1,n+1, 1,1,1,1,1,n)
  tke=q2; exm=0.; el=0.; rqc=0.; wmf=0.; lmf=0.; kpbl=0
  if(cid==12) then
    ct=.3; ru=.001; rv=-.002; rth=.0001; rqv=1.e-7; rqc=2.e-8
    ! QNSEPBL ignores these optional work arrays in its active source.
    wmf=.01; lmf=50.
  endif
  write(*,'(a)') 'INPUT'
  call dump()
  call qnsepbl(dt,stepbl,ht,dz,p,pint,th,t,exner,qv,qc,u,v,rho, &
               tsk,qsfc,chklowq,thz0,qz0,uz0,vz0,corf,lowlyr,xland,sice,snow, &
               tke,exh,exm,ustar,znt,el,pblh,kpbl,ct,akhs,akms,lh,hfx,lmf,wmf, &
               ru,rv,rth,rqv,rqc, &
               1,2,1,2,1,n+1, 1,1,1,1,1,n+1, 1,1,1,1,1,n)
  write(*,'(a)') 'OUTPUT'
  call dump()
contains
  subroutine dump_surface(outputs)
    logical, intent(in) :: outputs
    call scalar('scm_force_flux',real(scm_flag))
    call scalar('itimestep',real(itimestep)); call scalar('ht',ht(1,1))
    call scalar('tsk',tsk(1,1)); call scalar('xland',xland(1,1)); call scalar('ustar',ustar(1,1))
    call scalar('akhs',akhs(1,1)); call scalar('akms',akms(1,1)); call scalar('hfx',hfx(1,1))
    call scalar('qfx',qfx(1,1)); call scalar('flx_lh',lh(1,1)); call scalar('thz0',thz0(1,1))
    call scalar('qz0',qz0(1,1)); call scalar('uz0',uz0(1,1)); call scalar('vz0',vz0(1,1))
    call scalar('qsfc',qsfc(1,1)); call scalar('znt',znt(1,1)); call scalar('z0base',zbase(1,1))
    call scalar('pblh',pblh(1,1)); call scalar('mavail',mavail(1,1)); call scalar('psfc',pint(1,1,1))
    if(outputs) then
      call scalar('rmol',rmol(1,1)); call scalar('rib',rib(1,1)); call scalar('ct',ct(1,1))
      call scalar('chs',chs(1,1)); call scalar('chs2',chs2(1,1)); call scalar('cqs2',cqs2(1,1))
      call scalar('flhc',flhc(1,1)); call scalar('flqc',flqc(1,1)); call scalar('qgh',qgh(1,1))
      call scalar('cpm',cpm(1,1)); call scalar('u10',u10(1,1)); call scalar('v10',v10(1,1))
      call scalar('th02',th02(1,1)); call scalar('th10',th10(1,1)); call scalar('t02',t02(1,1))
      call scalar('tshltr',tshltr(1,1)); call scalar('q02',q02(1,1)); call scalar('qshltr',qshltr(1,1))
      call scalar('q10',q10(1,1)); call scalar('pshltr',pshltr(1,1)); call scalar('u10e',u10e(1,1))
      call scalar('v10e',v10e(1,1))
    endif
    call col('u',u); call col('v',v); call col('temperature',t); call col('theta',th)
    call col('qv',qv); call col('qc',qc); call col('p_mid',p); call col('dz',dz); call col('q2',q2)
  end subroutine
  subroutine scalar(name,a)
    character(*), intent(in) :: name
    real, intent(in) :: a
    write(*,'(a,a,es25.17)') name,'=',a
  end subroutine
  subroutine col(name,a)
    character(*), intent(in) :: name
    real, intent(in) :: a(1,n+1,1)
    integer :: kk,nn
    nn=n
    if(name=='p_int') nn=n+1
    do kk=1,nn
      write(*,'(a,a,i0,a,es25.17)') name,'[',kk,']=',a(1,kk,1)
    enddo
  end subroutine
  subroutine dump()
    call scalar('dt',dt); call scalar('stepbl',real(stepbl)); call scalar('ht',ht(1,1))
    call scalar('tsk',tsk(1,1)); call scalar('xland',xland(1,1)); call scalar('ustar',ustar(1,1))
    call scalar('akhs',akhs(1,1)); call scalar('akms',akms(1,1)); call scalar('hfx',hfx(1,1))
    call scalar('elflx',lh(1,1)); call scalar('thz0',thz0(1,1)); call scalar('qz0',qz0(1,1))
    call scalar('uz0',uz0(1,1)); call scalar('vz0',vz0(1,1)); call scalar('qsfc',qsfc(1,1))
    call scalar('ct',ct(1,1)); call scalar('chklowq',chklowq(1,1)); call scalar('corf',corf(1,1))
    call scalar('snow',snow(1,1)); call scalar('sice',sice(1,1))
    call scalar('PBLH',pblh(1,1)); call scalar('KPBL',real(kpbl(1,1)))
    call col('u',u); call col('v',v); call col('temperature',t); call col('theta',th)
    call col('qv',qv); call col('qc',qc); call col('p_mid',p); call col('p_int',pint)
    call col('exner',exner); call col('dz',dz); call col('tke',tke)
    call col('EXCH_H',exh); call col('EXCH_M',exm); call col('EL_MYJ',el)
    call col('RUBLTEN',ru); call col('RVBLTEN',rv); call col('RTHBLTEN',rth)
    call col('RQVBLTEN',rqv); call col('RQCBLTEN',rqc)
  end subroutine
end program
