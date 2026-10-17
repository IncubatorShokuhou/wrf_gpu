subroutine original_timescale(wind,vconv,tadvec,timec,nic,abe,sclvel)
implicit none
real,intent(in)::wind
real,intent(out)::vconv,tadvec,timec,abe,sclvel
integer,intent(out)::nic
integer,parameter::klcl=1,l5=2,ltop=3,lc=1,kpbl=2,i=1,j=1
real::wspd(3),u0(3),v0(3),p0(3),t0(3),q0(3),z0(3),cldhgt(3)
real::dx,dt,chmin,scale_fac,densplume,wst,hfx(1,1),cp,g,r,pblh(1,1)
real::thetav,eps1u,zlcl,zlcl_kay,ust(1,1),zol(1,1),frc2,zz_kay
u0=wind;v0=0.;p0=100000.;t0=300.;q0=.015;z0=800.;cldhgt=6000.
dx=9000.;dt=54.;chmin=4000.;scale_fac=2.;hfx=150.;cp=1004.5;g=9.81;r=287.
pblh=1000.;zlcl=500.;ust=.3;zol=-.1;abe=1000.
        WSPD(KLCL)=SQRT(U0(KLCL)*U0(KLCL)+V0(KLCL)*V0(KLCL))
        WSPD(L5)=SQRT(U0(L5)*U0(L5)+V0(L5)*V0(L5))
        WSPD(LTOP)=SQRT(U0(LTOP)*U0(LTOP)+V0(LTOP)*V0(LTOP))
        VCONV=.5*(WSPD(KLCL)+WSPD(L5))
!...for ETA model, DX is a function of location...
        TIMEC=DX/VCONV
        TADVEC=TIMEC
!
!ckay
!new dynTau based on subcloud layer scales : note Z0(KPBL)=altitude of source layer

         TIMEC = Amax1(CHMIN,CLDHGT(LC))
         TIMEC = TIMEC*Scale_Fac

!ckay SCLvel = SubCloudLayerVELOCITY = Wsb
!ckay estimate WSTAR to allow most of the PBL schemes here...
         densPlume = P0(1)/R/T0(1)
         WST = HFX(I,J)/densPlume/CP  ! hfx in kinematic units  CKAY
         WST = amax1(0.,WST)  ! +ve hfx only
         WST = G*PBLH(I,J)*WST
         thetav = (1.E5/P0(1))**(R/CP)
         thetav = T0(1) * thetav
         eps1u = 0.622
         thetav = thetav*(1.+Q0(1)*eps1u)
         WST = WST/thetav
         WST = WST**0.3333

         SCLvel = WST**3
         ZLCL_KAY = amax1(ZLCL,Z0(KPBL))
         SCLvel = SCLvel/PBLH(I,J)
         SCLvel = SCLvel*ZLCL_kay   
         SCLvel = SCLvel**0.333   ! Wsb=SubCloudLayerVelocity for ConvectivePBL
         if(ZOL(i,J).le.0.0) then
           FRC2=3.8*Ust(I,J)*Ust(I,J)
           FRC2 = FRC2 + 0.22*SCLvel*SCLvel
           zz_kay = -1.0*ZOL(I,j)
           ZLCL_KAY = zz_kay**(2./3.)
           ZLCL_KAY = ZLCL_KAY * (1.9*Ust(I,J)*Ust(I,J))
           FRC2 = FRC2 + ZLCL_KAY
         else
           FRC2=3.8*Ust(I,J)*Ust(I,J)
         end if 

          FRC2 = SQRT(FRC2)  
          SCLvel =  FRC2  ! Wsb=new subcloud layer velocity scale for all conditions
         IF(SCLvel.lt.0.1) SCLvel = 0.1
         if(ABE.le.0.0) ABE = 1.0 
         TIMEC = TIMEC/((0.03*SCLvel*ABE)**0.3333)

!ckay: this dynTau is good for the Deep as well as Shallow Cu clouds
        TIMEC = AMAX1(TADVEC, TIMEC)
 
         NIC=NINT(TIMEC/DT)
         TIMEC=FLOAT(NIC)*DT
         TIMEC=MIN(TIMEC,86400.)  !JRJ Ramboll: cap convective time scale at 24 hrs
end subroutine
