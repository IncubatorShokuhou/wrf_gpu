PROGRAM components_oracle
  USE module_sf_bep, ONLY: flux_flat, flux_wall, soil_temp, interpol, urban_meso, interp_length
  USE module_sf_bep_bem, ONLY: bem_flat=>flux_flat, bem_wall=>flux_wall
  USE module_sf_bem, ONLY: regtemp, reghum, air_cond
  IMPLICIT NONE
  INTEGER :: r,k,j,id,s,nd
  REAL :: dz,z0,u,v,theta,theta0,surface,rho,angle,dt,pressure,qv,window,drag
  REAL :: x(12),temp(10),depth(10),ala(10),cs(10),pt,rt,gf
  REAL :: sw,rl,em,alb,heat
  REAL :: z(6),zu(18),c(6),cu(18),prob(18),ss(18),bs(2),ws(2)
  REAL :: wall(4,18,7),flat(2,18,4),sf(6),vl(6),a(6,4),b(6,4),dlg(6),dl(6),dzz(6)
  REAL :: hsneed,hlneed,hsout,hlout,consump,qb,tlev,shum,vol,hour,hs,hl
  DO r=1,48
    dz=5.+MOD(r,5)*3.;z0=0.01+0.01*MOD(r,4)
    u=-8.+MOD(r,17);v=-5.+MOD(r,11)
    IF (r==1) THEN
      u=0.;v=0.
    ENDIF
    theta=280.+MOD(r,25);theta0=300.;surface=theta+REAL(MOD(r,7)-3)*3.
    rho=0.8+0.01*r;angle=REAL(MOD(r,8))*0.392699081625
    dt=6.*(1+MOD(r,4));IF (MOD(r,12)==0) dt=600.
    pressure=80000.+400.*r;qv=0.002+0.0001*r;window=surface+2.;drag=0.4
    x=(/dz,z0,u,v,theta,theta0,surface,rho,angle,dt,pressure,qv/)
    DO j=1,12
      CALL emit(0,r,j,0,x(j))
    ENDDO
    CALL flux_flat(dz,z0,u,v,theta,theta0,surface,x(1),x(2),x(3),x(4))
    DO j=1,4
      CALL emit(1,r,j,0,x(j))
    ENDDO
    CALL flux_wall(u,v,theta,rho,surface,x(1),x(2),x(3),x(4),x(5),x(6),x(7),angle,dt)
    DO j=1,7
      CALL emit(2,r,j,0,x(j))
    ENDDO
    CALL bem_flat(dz,z0,u,v,theta,theta0,surface,qv,x(1),x(2),x(3),x(4),x(5),rho,pressure)
    DO j=1,5
      CALL emit(3,r,j,0,x(j))
    ENDDO
    CALL bem_wall(u,v,theta,rho,surface,window,x(1),x(2),x(3),x(4),x(5),x(6),x(7),angle,dt,drag)
    DO j=1,7
      CALL emit(4,r,j,0,x(j))
    ENDDO
    DO k=1,10
      temp(k)=theta+0.2*k;depth(k)=0.01+0.005*k
      ala(k)=0.0000004+0.00000001*k;cs(k)=1500000.+10000.*k
    ENDDO
    sw=REAL(MOD(r,3))*400.;rl=250.+r;em=0.9;alb=0.2;heat=REAL(MOD(r,9)-4)*20.
    DO k=1,10
      CALL emit(10,r,k,0,temp(k));CALL emit(10,r,10+k,0,depth(k))
      CALL emit(10,r,20+k,0,ala(k));CALL emit(10,r,30+k,0,cs(k))
    ENDDO
    x(1:5)=(/sw,rl,em,alb,heat/)
    DO j=1,5
      CALL emit(10,r,40+j,0,x(j))
    ENDDO
    DO s=1,6
      CALL soil_temp(10,depth,temp,pt,ala,cs,sw,rl,pressure,dt,em,alb,rt,heat,gf)
      DO k=1,10
        CALL emit(5,r,k,s,temp(k))
      ENDDO
      CALL emit(5,r,11,s,pt);CALL emit(5,r,12,s,rt);CALL emit(5,r,13,s,gf)
    ENDDO
    ! Deliberately non-aligned urban/mesoscale interfaces exercise overlap rules.
    z=(/0.,3.,8.,15.,30.,60./)
    DO k=1,18
      zu(k)=5.*(k-1)
    ENDDO
    DO k=1,6
      c(k)=theta+0.25*k;dzz(k)=1.
    ENDDO
    DO k=1,5
      dzz(k)=z(k+1)-z(k)
    ENDDO
    cu=0.
    CALL interpol(1,6,1,5,6,z,zu,c,cu)
    DO k=1,6
      CALL emit(6,r,k,0,cu(k))
    ENDDO
    prob=0.;ss=0.;ss(3)=0.2;ss(4)=0.3;ss(5)=0.5
    prob(1:3)=1.;prob(4)=0.8;prob(5)=0.5
    bs=(/10.,20./);ws=(/15.,12./);nd=1+MOD(r,2)
    DO id=1,4
      DO k=1,18
        DO j=1,7
          wall(id,k,j)=REAL(MOD(id+k+j+r,7)-3)*0.001*j
        ENDDO
      ENDDO
    ENDDO
    DO id=1,2
      DO k=1,18
        DO j=1,4
          flat(id,k,j)=REAL(MOD(id+k+j+r,5)-2)*0.01*j
        ENDDO
      ENDDO
    ENDDO
    DO k=1,6
      CALL emit(11,r,k,0,z(k));CALL emit(11,r,6+k,0,c(k))
    ENDDO
    DO k=1,18
      CALL emit(11,r,12+k,0,zu(k));CALL emit(11,r,30+k,0,prob(k))
      CALL emit(11,r,48+k,0,ss(k))
      DO id=1,4
        DO j=1,7
          CALL emit(12,r,(id-1)*18*7+(k-1)*7+j,0,wall(id,k,j))
        ENDDO
      ENDDO
      DO id=1,2
        DO j=1,4
          CALL emit(13,r,(id-1)*18*4+(k-1)*4+j,0,flat(id,k,j))
        ENDDO
      ENDDO
    ENDDO
    a=0.;b=0.;sf=0.;vl=0.;dlg=0.;dl=0.
    CALL urban_meso(nd,1,6,1,5,6,z,dzz,zu,prob,ss,bs,ws,sf,vl, &
      wall(:,:,1),wall(:,:,2),wall(:,:,3),wall(:,:,4),wall(:,:,5),wall(:,:,6),wall(:,:,7), &
      flat(:,:,1),flat(:,:,2),flat(:,:,3),flat(:,:,4),a(:,1),a(:,2),a(:,3),a(:,4), &
      b(:,1),b(:,2),b(:,3),b(:,4))
    CALL interp_length(nd,1,6,1,5,6,zu,z,ss,ws,bs,dlg,dl)
    DO k=1,5
      DO j=1,4
        CALL emit(7,r,10*(k-1)+j,0,a(k,j))
        CALL emit(7,r,10*(k-1)+4+j,0,b(k,j))
      ENDDO
      CALL emit(7,r,10*(k-1)+9,0,sf(k));CALL emit(7,r,10*(k-1)+10,0,vl(k))
      CALL emit(8,r,2*k-1,0,dlg(k));CALL emit(8,r,2*k,0,dl(k))
    ENDDO
    CALL emit(7,r,51,0,sf(6))
    ! AC hours include inclusive boundary, disabled, inactive, comfort and rate caps.
    hour=REAL(MOD(r,25));qb=100000.+1000.*r;hs=REAL(MOD(r,9)-4)*100.
    vol=50.+r;tlev=298.+REAL(MOD(r,7)-3)*0.6
    shum=0.005+REAL(MOD(r,7)-3)*0.0003;hl=REAL(MOD(r,9)-4)*50.
    x(1:7)=(/hour,qb,hs,vol,tlev,shum,hl/)
    DO j=1,7
      CALL emit(14,r,j,0,x(j))
    ENDDO
    CALL regtemp(MOD(r,2),hour,dt,qb,hs,tlev,8.,20.,298.,0.5,hsneed)
    CALL reghum(MOD(r,2),hour,dt,vol,rho,2450000.,hl,shum,8.,20.,0.005,0.0002,hlneed)
    CALL emit(9,r,1,0,hsneed);CALL emit(9,r,2,0,hlneed)
    CALL air_cond(hsneed,hlneed,dt,hsout,hlout,consump,3.5)
    CALL emit(9,r,3,0,hsout);CALL emit(9,r,4,0,hlout)
    CALL emit(9,r,5,0,consump);CALL emit(9,r,6,0,hlneed)
  ENDDO
CONTAINS
  SUBROUTINE emit(kind,regime,field,step,value)
    INTEGER :: kind,regime,field,step
    REAL :: value
    WRITE(*,'(4(I0,1X),ES24.16E3)') kind,regime,field,step,value
  END SUBROUTINE
END PROGRAM
