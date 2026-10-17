! Driver only: both modules under test are compiled from unmodified pristine WRF.
program oracle
  use module_sf_sfcdiags_ruclsm
  use module_sf_ruclsm, only: ruclsminit
  implicit none
  integer, parameter :: n=12
  real :: hfx(n,1),qfx(n,1),tsk(n,1),qsfc(n,1),cqs(n,1),cqs2(n,1),chs(n,1),chs2(n,1)
  real :: t2(n,1),th2(n,1),q2(n,1),psfc(n,1),snow(n,1)
  real :: t3(n,1,1),qv(n,1,1),rho(n,1,1),p(n,1,1)
  real :: tslb(n,9,1),smois(n,9,1),sh2o(n,9,1),smfr(n,9,1),xice(n,1),mavail(n,1),znt(n,1)
  integer :: ivg(n,1),isl(n,1),i,k,nzs,lu
  character(len=24) :: landuse
  do i=1,n
    tsk(i,1)=250.+4.*i
    t3(i,1,1)=tsk(i,1)+(-1.)**i*5.
    qv(i,1,1)=.001*i
    rho(i,1,1)=.8+.04*i
    p(i,1,1)=70000.+2000.*i
    psfc(i,1)=p(i,1,1)+500.
    hfx(i,1)=(-1.)**i*30.*i
    qfx(i,1)=(-1.)**i*.00002*i
    qsfc(i,1)=.0015*i
    cqs(i,1)=.005+.001*i
    cqs2(i,1)=.003+.001*i
    chs2(i,1)=.004+.001*i
  enddo
  chs2(1,1)=0.; cqs2(2,1)=0.
  chs2(3,1)=.000009; cqs2(4,1)=.000009
  chs2(5,1)=.00001; cqs2(6,1)=.00001
  chs=0.; snow=0.; t2=0.; th2=0.; q2=0.
  call SFCDIAGS_RUCLSM(hfx,qfx,tsk,qsfc,cqs,cqs2,chs,chs2,t2,th2,q2, &
      t3,qv,rho,p,psfc,snow,1004.5,287.,287./1004.5, &
      1,n+1,1,2,1,2,1,n,1,1,1,1,1,n,1,1,1,1)
  do i=1,n
    write(*,'(A,15(1X,ES25.17E3))') 'DIAG', &
      hfx(i,1),qfx(i,1),tsk(i,1),qsfc(i,1),cqs(i,1),cqs2(i,1),chs2(i,1), &
      t3(i,1,1),qv(i,1,1),rho(i,1,1),p(i,1,1),psfc(i,1),t2(i,1),th2(i,1),q2(i,1)
  enddo
  do lu=1,2
    if(lu==1) then
      landuse='USGS'
    else
      landuse='MODIFIED_IGBP_MODIS_NOAH'
    endif
    do nzs=6,9,3
      do i=1,n
        ivg(i,1)=7; isl(i,1)=3; xice(i,1)=0.
        if(i==n-1) isl(i,1)=14
        if(i==n) xice(i,1)=1.
        do k=1,nzs
          tslb(i,k,1)=260.+i+k
          smois(i,k,1)=.1+.02*i+.001*k
        enddo
      enddo
      sh2o=-99.; smfr=-99.; mavail=-99.; znt=-99.
      call ruclsminit(sh2o,smfr,tslb,smois,isl,ivg,trim(landuse),xice,mavail,nzs,16,24, &
          znt,.false.,.true.,1,n+1,1,2,1,2,1,n,1,1,1,1,1,n,1,1,1,1)
      do i=1,n
        do k=1,nzs
          write(*,'(A,5(1X,I3),7(1X,ES25.17E3))') 'INIT',lu,nzs,i,k,isl(i,1), &
            tslb(i,k,1),smois(i,k,1),xice(i,1),sh2o(i,k,1),smfr(i,k,1),mavail(i,1),znt(i,1)
        enddo
      enddo
    enddo
  enddo
end program
