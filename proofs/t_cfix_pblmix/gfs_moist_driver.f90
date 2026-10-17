! =====================================================================
! t-cfix-pblmix: GFS PBL (bl_pbl_physics=3) public-call oracle on REAL
! cloudy CPU-WRF columns, condensate wiring as module_pbl_driver.F:1405-1420
! (QC3D=qc_curr, QI3D=qi_curr, RQCBLTEN, RQIBLTEN, P_QI/P_FIRST_SCALAR).
!
! Drives the UNMODIFIED pristine WRF module_bl_gfs.F:BL_GFS on the operands
! written by real_columns.py (one i-row per column, one j-row) and dumps
! the WRF tendencies.  argv(1) = input text, argv(2) = P_QI present (T/F).
! =====================================================================
PROGRAM gfs_moist_oracle
  USE module_bl_gfs, ONLY : bl_gfs
  IMPLICIT NONE
  REAL, PARAMETER :: G=9.81, R_D=287.0, CP=7.0*R_D/2.0, R_V=461.6, XLV=2.5E6
  REAL, PARAMETER :: ROVCP=R_D/CP, ROVG=R_D/G, EP1=R_V/R_D-1.0, EP2=R_D/R_V
  REAL, PARAMETER :: KARMAN=0.4, P1000=1.0E5, SVP1=0.6112, SVP2=17.67, SVP3=29.65
  REAL, PARAMETER :: SVPT0=273.15, EOMEG=7.2921E-5, STBOLT=5.67051E-8, DT=54.0
  INTEGER :: p_qi, p_first
  INTEGER :: ncol, kx, c, k, ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: U,V,T,TH,QV,QC,QI,P,PII,DZ,ZF
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: RUB,RVB,RTHB,RQVB,RQCB,RQIB
  REAL, ALLOCATABLE, DIMENSION(:,:) :: PSFC,ZNT,UST,ZOL,HOL,PBL,PSIM,PSIH,XLAND,HFX,QFX,TSK
  REAL, ALLOCATABLE, DIMENSION(:,:) :: GZ1OZ0,WSPD,BR,REGIME
  INTEGER, ALLOCATABLE, DIMENSION(:,:) :: KPBL2D
  REAL :: s(12), row(11)
  CHARACTER(LEN=512) :: path, fq
  LOGICAL :: flag_qi

  CALL GET_COMMAND_ARGUMENT(1, path)
  CALL GET_COMMAND_ARGUMENT(2, fq)
  flag_qi = (TRIM(fq) == 'T')
  OPEN(10, FILE=TRIM(path), STATUS='OLD', ACTION='READ')
  READ(10,*) ncol, kx
  ids=1; ide=ncol+1; jds=1; jde=2; kds=1; kde=kx+1
  ims=1; ime=ncol; jms=1; jme=1; kms=1; kme=kx+1
  its=1; ite=ncol; jts=1; jte=1; kts=1; kte=kx
  ALLOCATE(U(ims:ime,kms:kme,jms:jme))
  ALLOCATE(V(ims:ime,kms:kme,jms:jme))
  ALLOCATE(T(ims:ime,kms:kme,jms:jme))
  ALLOCATE(TH(ims:ime,kms:kme,jms:jme))
  ALLOCATE(QV(ims:ime,kms:kme,jms:jme))
  ALLOCATE(QC(ims:ime,kms:kme,jms:jme))
  ALLOCATE(QI(ims:ime,kms:kme,jms:jme))
  ALLOCATE(P(ims:ime,kms:kme,jms:jme))
  ALLOCATE(PII(ims:ime,kms:kme,jms:jme))
  ALLOCATE(DZ(ims:ime,kms:kme,jms:jme))
  ALLOCATE(ZF(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RUB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RVB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RTHB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RQVB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RQCB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RQIB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(PSFC(ims:ime,jms:jme))
  ALLOCATE(ZNT(ims:ime,jms:jme))
  ALLOCATE(UST(ims:ime,jms:jme))
  ALLOCATE(ZOL(ims:ime,jms:jme))
  ALLOCATE(HOL(ims:ime,jms:jme))
  ALLOCATE(PBL(ims:ime,jms:jme))
  ALLOCATE(PSIM(ims:ime,jms:jme))
  ALLOCATE(PSIH(ims:ime,jms:jme))
  ALLOCATE(XLAND(ims:ime,jms:jme))
  ALLOCATE(HFX(ims:ime,jms:jme))
  ALLOCATE(QFX(ims:ime,jms:jme))
  ALLOCATE(TSK(ims:ime,jms:jme))
  ALLOCATE(GZ1OZ0(ims:ime,jms:jme))
  ALLOCATE(WSPD(ims:ime,jms:jme))
  ALLOCATE(BR(ims:ime,jms:jme))
  ALLOCATE(REGIME(ims:ime,jms:jme))
  ALLOCATE(KPBL2D(ims:ime,jms:jme))
  U=0.;V=0.;T=0.;TH=0.;QV=0.;QC=0.;QI=0.;P=0.;PII=0.;DZ=0.;ZF=0.
  DO c = 1, ncol
    READ(10,*) s
    PSFC(c,1)=s(1); ZNT(c,1)=s(2); UST(c,1)=s(3); HFX(c,1)=s(4); QFX(c,1)=s(5); TSK(c,1)=s(6)
    XLAND(c,1)=s(7); WSPD(c,1)=s(8); BR(c,1)=s(9); PSIM(c,1)=s(10); PSIH(c,1)=s(11); GZ1OZ0(c,1)=s(12)
    DO k = 1, kx
      READ(10,*) row   ! U V T TH QV QC QI P PI DZ Z
      U(c,k,1)=row(1); V(c,k,1)=row(2); T(c,k,1)=row(3); TH(c,k,1)=row(4); QV(c,k,1)=row(5)
      QC(c,k,1)=row(6); QI(c,k,1)=row(7); P(c,k,1)=row(8); PII(c,k,1)=row(9); DZ(c,k,1)=row(10); ZF(c,k,1)=row(11)
    END DO
  END DO
  CLOSE(10)
  RUB=0.;RVB=0.;RTHB=0.;RQVB=0.;RQCB=0.;RQIB=0.;ZOL=0.;HOL=0.;PBL=0.;KPBL2D=0;REGIME=0.

  p_first = 2
  p_qi = 1
  IF (flag_qi) p_qi = 3
  CALL bl_gfs(U3D=U,V3D=V,TH3D=TH,T3D=T,QV3D=QV,QC3D=QC,QI3D=QI,P3D=P,PI3D=PII, &
              RUBLTEN=RUB,RVBLTEN=RVB,RTHBLTEN=RTHB,RQVBLTEN=RQVB,RQCBLTEN=RQCB,RQIBLTEN=RQIB, &
              CP=CP,G=G,ROVCP=ROVCP,R=R_D,ROVG=ROVG,P_QI=p_qi,P_FIRST_SCALAR=p_first, &
              dz8w=DZ,z=ZF,PSFC=PSFC,UST=UST,PBL=PBL,PSIM=PSIM,PSIH=PSIH, &
              HFX=HFX,QFX=QFX,TSK=TSK,GZ1OZ0=GZ1OZ0,WSPD=WSPD,BR=BR, &
              DT=DT,KPBL2D=KPBL2D,EP1=EP1,KARMAN=KARMAN, &
              ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde,ims=ims,ime=ime,jms=jms,jme=jme, &
              kms=kms,kme=kme,its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte)

  WRITE(*,'(A,I0,1X,I0)') 'NCOL_KX ', ncol, kx
  DO c = 1, ncol
    WRITE(*,'(A,I0,1X,ES26.16E3,1X,I0)') 'PBL ', c, PBL(c,1), KPBL2D(c,1)
    WRITE(*,'(A,I0,1X,ES26.16E3)') 'UST ', c, UST(c,1)
    DO k = 1, kx
      WRITE(*,'(A,I0,1X,I0,6(1X,ES26.16E3))') 'TEN ', c, k, RUB(c,k,1), RVB(c,k,1), RTHB(c,k,1), &
            RQVB(c,k,1), RQCB(c,k,1), RQIB(c,k,1)
    END DO
  END DO
END PROGRAM gfs_moist_oracle
