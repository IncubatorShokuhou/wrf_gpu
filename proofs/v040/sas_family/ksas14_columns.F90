! Pristine CU_KSAS batch boundary. N and KX arbitrary; shared domain ZNU.
PROGRAM ksas14_columns
  USE module_cu_ksas, ONLY: cu_ksas
  IMPLICIT NONE
  REAL, PARAMETER :: G=9.81,RD=287.,RV=461.6,CP=7.*RD/2.,XLV=2.5E6,XLS=2.85E6
  REAL, PARAMETER :: CLIQ=4190.,CPV=1870.,CICE=2106.,PSAT=610.78,EP1=RV/RD-1.,EP2=RD/RV
  INTEGER :: N,KX,I,K,STEPCU,MP
  REAL :: DT,DX
  REAL, ALLOCATABLE :: T(:,:,:),QV(:,:,:),QC(:,:,:),QI(:,:,:),P(:,:,:),PII(:,:,:)
  REAL, ALLOCATABLE :: RHO(:,:,:),DZ(:,:,:),U(:,:,:),V(:,:,:),W(:,:,:),P8W(:,:,:)
  REAL, ALLOCATABLE :: RTH(:,:,:),RQV(:,:,:),RQC(:,:,:),RQI(:,:,:),RU(:,:,:),RVT(:,:,:)
  REAL, ALLOCATABLE :: RN(:,:),PRATE(:,:),HBOT(:,:),HTOP(:,:),XLAND(:,:),HPBL(:,:),HOLD(:,:)
  REAL, ALLOCATABLE :: HFX(:,:),QFX(:,:),ZNU(:)
  LOGICAL, ALLOCATABLE :: ACTIVE(:,:)
  LOGICAL :: FQC,FQI
  READ(*,*) N,KX
  READ(*,*) DT,STEPCU,MP,DX
  ALLOCATE(T(N,KX+1,1),QV(N,KX+1,1),QC(N,KX+1,1),QI(N,KX+1,1),P(N,KX+1,1),PII(N,KX+1,1))
  ALLOCATE(RHO(N,KX+1,1),DZ(N,KX+1,1),U(N,KX+1,1),V(N,KX+1,1),W(N,KX+1,1),P8W(N,KX+1,1))
  ALLOCATE(RTH(N,KX+1,1),RQV(N,KX+1,1),RQC(N,KX+1,1),RQI(N,KX+1,1),RU(N,KX+1,1),RVT(N,KX+1,1))
  ALLOCATE(RN(N,1),PRATE(N,1),HBOT(N,1),HTOP(N,1),XLAND(N,1),HPBL(N,1),HOLD(N,1))
  ALLOCATE(HFX(N,1),QFX(N,1),ZNU(KX+1),ACTIVE(N,1))
  T=0.;QV=0.;QC=0.;QI=0.;P=0.;PII=1.;RHO=0.;DZ=0.;U=0.;V=0.;W=0.;P8W=0.
  RTH=0.;RQV=0.;RQC=0.;RQI=0.;RU=0.;RVT=0.;RN=0.;PRATE=0.;HBOT=0.;HTOP=0.
  HFX=0.;QFX=0.;ZNU=0.;ACTIVE=.TRUE.
  READ(*,*) (ZNU(K),K=1,KX)
  DO I=1,N
    READ(*,*) XLAND(I,1),HPBL(I,1),HOLD(I,1)
    READ(*,*) (T(I,K,1),K=1,KX)
    READ(*,*) (QV(I,K,1),K=1,KX)
    READ(*,*) (QC(I,K,1),K=1,KX)
    READ(*,*) (QI(I,K,1),K=1,KX)
    READ(*,*) (P(I,K,1),K=1,KX)
    READ(*,*) (PII(I,K,1),K=1,KX)
    READ(*,*) (RHO(I,K,1),K=1,KX)
    READ(*,*) (DZ(I,K,1),K=1,KX)
    READ(*,*) (U(I,K,1),K=1,KX)
    READ(*,*) (V(I,K,1),K=1,KX)
    READ(*,*) (P8W(I,K,1),K=1,KX+1)
    READ(*,*) (W(I,K,1),K=1,KX+1)
  END DO
  FQC=MP/=0
  FQI=MP/=0.AND.MP/=1.AND.MP/=3
  CALL cu_ksas(dt=DT,dx=DX,p3di=P8W,p3d=P,pi3d=PII,qc3d=QC,qi3d=QI,rho3d=RHO, &
    itimestep=1,stepcu=STEPCU,hbot=HBOT,htop=HTOP,cu_act_flag=ACTIVE, &
    rthcuten=RTH,rqvcuten=RQV,rqccuten=RQC,rqicuten=RQI,rucuten=RU,rvcuten=RVT, &
    qv3d=QV,t3d=T,raincv=RN,pratec=PRATE,xland=XLAND,dz8w=DZ,w=W,u3d=U,v3d=V, &
    hpbl=HPBL,hpbl_hold=HOLD,hfx=HFX,qfx=QFX,znu=ZNU,mp_physics=MP,dx_factor_nsas=0, &
    p_qc=2,p_qi=3,p_first_scalar=2,pgcon=.55,cp=CP,cliq=CLIQ,cpv=CPV,g=G,xlv=XLV, &
    r_d=RD,r_v=RV,ep_1=EP1,ep_2=EP2,cice=CICE,xls=XLS,psat=PSAT,f_qi=FQI,f_qc=FQC, &
    ids=1,ide=N+1,jds=1,jde=2,kds=1,kde=KX+1,ims=1,ime=N,jms=1,jme=1,kms=1,kme=KX+1, &
    its=1,ite=N,jts=1,jte=1,kts=1,kte=KX)
  DO I=1,N
    WRITE(*,'(I7,4ES27.17E3)') I,RN(I,1),PRATE(I,1),HBOT(I,1),HTOP(I,1)
    WRITE(*,'(*(ES27.17E3))') (RTH(I,K,1),K=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RQV(I,K,1),K=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RQC(I,K,1),K=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RQI(I,K,1),K=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RU(I,K,1),K=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RVT(I,K,1),K=1,KX)
  END DO
END PROGRAM ksas14_columns
