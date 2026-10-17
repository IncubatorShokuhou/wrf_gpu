! Single-column WRF oracle. Compile the UNMODIFIED module_sf_lake.F with
! EM_CORE=1, REAL=4; Lake performs the same REAL->r8 seam as surface_driver.
program lake_column_oracle
  use module_sf_lake
  implicit none
  integer :: veg(1,1), soil(1,1), depthflag, maskflag, use_depth, nsteps, step, k
  real :: ht(1,1), snow(1,1), xland(1,1), xice(1,1), mask(1,1), skin(1,1)
  real :: min_elev, default_depth, depth(1,1), dep(1,1), tke(1,1), snowdp(1,1)
  real :: swe(1,1), nsnow(1,1), ground(1,1)
  real :: tl(1,10,1), ice(1,10,1), zl(1,10,1), dl(1,10,1)
  real :: ts(1,-4:10,1), wi(1,-4:10,1), wl(1,-4:10,1), wv(1,-4:10,1)
  real :: zs(1,-4:10,1), ds(1,-4:10,1), zis(1,-5:10,1)
  real :: ws(1,10,1), cs(1,10,1), tm(1,10,1), td(1,10,1), tk(1,10,1)
  logical :: lakepoint(1,1)
  real :: ta(1,2,1), p(1,2,1), dzair(1,2,1), qv(1,2,1), u(1,2,1), v(1,2,1)
  real :: lw(1,1), emiss(1,1), rain(1,1), dt, sw(1,1), alb(1,1), lat(1,1)
  real :: sh(1,1), lh(1,1), gh(1,1), evap(1,1), t2(1,1), th2(1,1), q2(1,1)
  real(r8) :: es, est, qs, qst, temp, pres
  ! Input ABI is deliberately all REAL32 at the WRF caller boundary.
  read(*,*) veg, soil, ht, snow, skin, mask, xice, xland, depth
  read(*,*) maskflag, depthflag, use_depth, default_depth, min_elev, nsteps
  read(*,*) ta(1,1,1), p(1,1,1), p(1,2,1), dzair(1,1,1), qv(1,1,1), &
            u(1,1,1), v(1,1,1), lw, emiss, rain, dt, sw, alb, lat
  ta(1,2,1)=ta(1,1,1); dzair(1,2,1)=dzair(1,1,1)
  qv(1,2,1)=qv(1,1,1); u(1,2,1)=u(1,1,1); v(1,2,1)=v(1,1,1)
  ! Seed every output so undefined/non-lake slots can be distinguished from
  ! actual WRF assignments. No uninitialized oracle operands (E11/E83).
  dep=-777.; tke=-777.; snowdp=-777.; swe=-777.; nsnow=-777.; ground=-777.
  tl=-777.; ice=-777.; zl=-777.; dl=-777.; ts=-777.; wi=-777.; wl=-777.
  wv=-777.; zs=-777.; ds=-777.; zis=-777.; ws=-777.; cs=-777.; tm=-777.
  td=-777.; tk=-777.; lakepoint=.false.
  sh=-777.; lh=-777.; gh=-777.; evap=-777.; t2=-777.; th2=-777.; q2=-777.
  call lakeini(veg,soil,ht,snow,min_elev,.false.,default_depth,depth, &
       dep,tke,snowdp,swe,nsnow,ground,tl,ice,zl,dl,ts,wi,wl,wv,zs,ds,zis, &
       ws,cs,tm,17,xice,0.5,xland,skin,mask,maskflag,depthflag,use_depth, &
       td,tk,lakepoint,1,1,1,1,1,1,1,1)
  call dump_state(0)
  do step=1,nsteps
    call Lake(ta,p,dzair,qv,u,v,lw,emiss,rain,dt,sw,alb,lat,zl,dl,dep, &
         ws,cs,tm,td,tk,veg,ht,xland,17,xice,0.5,min_elev, &
         1,2,1,2,1,2,1,1,1,1,1,2,1,1,1,1,1,1, &
         swe,snowdp,nsnow,zs,ds,zis,wv,wl,wi,ground,ts,tl,tke,ice,mask, &
         sh,lh,gh,skin,evap,t2,th2,q2)
    call dump_state(step)
  enddo
  ! QSat's own explicit DOUBLE interface; temperatures cover both branches
  ! and its source-defined polynomial range limits.
  do k=1,8
    temp=150._r8+35._r8*k; pres=90000._r8
    call QSat(temp,pres,es,est,qs,qst)
    write(*,'(A,6(1X,ES26.17E3))') 'QSAT',temp,pres,es,est,qs,qst
  enddo
contains
  subroutine dump_state(n)
    integer, intent(in) :: n
    write(*,'(A,1X,I0)') 'STEP',n
    write(*,'(A,7(1X,ES26.17E3))') 'meta',real(veg(1,1)),real(soil(1,1)), &
         xland(1,1),xice(1,1),mask(1,1),real(depthflag),real(merge(1,0,lakepoint(1,1)))
    write(*,'(A,*(1X,ES26.17E3))') 'lakedepth2d',dep
    write(*,'(A,*(1X,ES26.17E3))') 'savedtke12d',tke
    write(*,'(A,*(1X,ES26.17E3))') 'snowdp2d',snowdp
    write(*,'(A,*(1X,ES26.17E3))') 'h2osno2d',swe
    write(*,'(A,*(1X,ES26.17E3))') 'snl2d',nsnow
    write(*,'(A,*(1X,ES26.17E3))') 't_grnd2d',ground
    write(*,'(A,*(1X,ES26.17E3))') 't_lake3d',tl
    write(*,'(A,*(1X,ES26.17E3))') 'lake_icefrac3d',ice
    write(*,'(A,*(1X,ES26.17E3))') 'z_lake3d',zl
    write(*,'(A,*(1X,ES26.17E3))') 'dz_lake3d',dl
    write(*,'(A,*(1X,ES26.17E3))') 't_soisno3d',ts
    write(*,'(A,*(1X,ES26.17E3))') 'h2osoi_ice3d',wi
    write(*,'(A,*(1X,ES26.17E3))') 'h2osoi_liq3d',wl
    write(*,'(A,*(1X,ES26.17E3))') 'h2osoi_vol3d',wv
    write(*,'(A,*(1X,ES26.17E3))') 'z3d',zs
    write(*,'(A,*(1X,ES26.17E3))') 'dz3d',ds
    write(*,'(A,*(1X,ES26.17E3))') 'zi3d',zis
    write(*,'(A,*(1X,ES26.17E3))') 'watsat3d',ws
    write(*,'(A,*(1X,ES26.17E3))') 'csol3d',cs
    write(*,'(A,*(1X,ES26.17E3))') 'tkmg3d',tm
    write(*,'(A,*(1X,ES26.17E3))') 'tkdry3d',td
    write(*,'(A,*(1X,ES26.17E3))') 'tksatu3d',tk
    write(*,'(A,*(1X,ES26.17E3))') 'flux',sh,lh,gh,skin,evap,t2,th2,q2,alb
  end subroutine dump_state
end program lake_column_oracle
