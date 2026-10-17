! =====================================================================
! t-cfix-pblmix: MYJ PBL (bl_pbl_physics=2) public-call oracle on REAL
! cloudy CPU-WRF columns, condensate wiring as module_pbl_driver.F:1446-1460
! (QCW=qc_curr, QCI=qi_curr -> flQI, RQCBLTEN, RQIBLTEN).
!
! Drives the UNMODIFIED pristine WRF module_bl_myjpbl.F:MYJPBL on the
! operands written by myj_columns.py (one i-row per column) and dumps the
! WRF tendencies.  argv(1) = input text, argv(2) = QCI present (T/F).
! =====================================================================
PROGRAM myj_moist_oracle
  USE module_bl_myjpbl, ONLY : myjpbl
  IMPLICIT NONE
  REAL, PARAMETER :: DT=54.0
  INTEGER :: ncol, kx, c, k, ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: U,V,T,TH,QV,QC,QI,P,PII,DZ,ZF,TKE,PINT,RHO,EXCH_H,EL
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: RUB,RVB,RTHB,RQVB,RQCB,RQIB
  REAL, ALLOCATABLE, DIMENSION(:,:) :: PSFC,ZNT,UST,HFX,QFX,TSK,XLAND,WSPD,BR,PSIM,PSIH,GZ1OZ0
  REAL, ALLOCATABLE, DIMENSION(:,:) :: HT,LH,SNOW,SICE,AKHS,AKMS,THZ0,QZ0,QSFC,CHKLOWQ,CT,UZ0,VZ0
  REAL, ALLOCATABLE, DIMENSION(:,:) :: PBLH,MIXHT
  INTEGER, ALLOCATABLE, DIMENSION(:,:) :: LOWLYR,KPBL
  REAL :: s(25), row(12)
  CHARACTER(LEN=512) :: path, fq

  CALL GET_COMMAND_ARGUMENT(1, path)
  CALL GET_COMMAND_ARGUMENT(2, fq)
  OPEN(10, FILE=TRIM(path), STATUS='OLD', ACTION='READ')
  READ(10,*) ncol, kx
  ids=1; ide=ncol+1; jds=1; jde=2; kds=1; kde=kx+1
  ims=1; ime=ncol; jms=1; jme=1; kms=1; kme=kx+1
  its=1; ite=ncol; jts=1; jte=1; kts=1; kte=kx
  ALLOCATE(U(ims:ime,kms:kme,jms:jme), V(ims:ime,kms:kme,jms:jme), T(ims:ime,kms:kme,jms:jme))
  ALLOCATE(TH(ims:ime,kms:kme,jms:jme), QV(ims:ime,kms:kme,jms:jme), QC(ims:ime,kms:kme,jms:jme))
  ALLOCATE(QI(ims:ime,kms:kme,jms:jme), P(ims:ime,kms:kme,jms:jme), PII(ims:ime,kms:kme,jms:jme))
  ALLOCATE(DZ(ims:ime,kms:kme,jms:jme), ZF(ims:ime,kms:kme,jms:jme), TKE(ims:ime,kms:kme,jms:jme))
  ALLOCATE(PINT(ims:ime,kms:kme,jms:jme), RHO(ims:ime,kms:kme,jms:jme), EXCH_H(ims:ime,kms:kme,jms:jme))
  ALLOCATE(EL(ims:ime,kms:kme,jms:jme), RUB(ims:ime,kms:kme,jms:jme), RVB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RTHB(ims:ime,kms:kme,jms:jme), RQVB(ims:ime,kms:kme,jms:jme), RQCB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(RQIB(ims:ime,kms:kme,jms:jme))
  ALLOCATE(PSFC(ims:ime,jms:jme), ZNT(ims:ime,jms:jme), UST(ims:ime,jms:jme), HFX(ims:ime,jms:jme))
  ALLOCATE(QFX(ims:ime,jms:jme), TSK(ims:ime,jms:jme), XLAND(ims:ime,jms:jme), WSPD(ims:ime,jms:jme))
  ALLOCATE(BR(ims:ime,jms:jme), PSIM(ims:ime,jms:jme), PSIH(ims:ime,jms:jme), GZ1OZ0(ims:ime,jms:jme))
  ALLOCATE(HT(ims:ime,jms:jme), LH(ims:ime,jms:jme), SNOW(ims:ime,jms:jme), SICE(ims:ime,jms:jme))
  ALLOCATE(AKHS(ims:ime,jms:jme), AKMS(ims:ime,jms:jme), THZ0(ims:ime,jms:jme), QZ0(ims:ime,jms:jme))
  ALLOCATE(QSFC(ims:ime,jms:jme), CHKLOWQ(ims:ime,jms:jme), CT(ims:ime,jms:jme), UZ0(ims:ime,jms:jme))
  ALLOCATE(VZ0(ims:ime,jms:jme), PBLH(ims:ime,jms:jme), MIXHT(ims:ime,jms:jme))
  ALLOCATE(LOWLYR(ims:ime,jms:jme), KPBL(ims:ime,jms:jme))
  U=0.;V=0.;T=0.;TH=0.;QV=0.;QC=0.;QI=0.;P=0.;PII=1.;DZ=0.;ZF=0.;TKE=0.;PINT=0.;RHO=1.
  DO c = 1, ncol
    READ(10,*) s
    PSFC(c,1)=s(1); ZNT(c,1)=s(2); UST(c,1)=s(3); HFX(c,1)=s(4); QFX(c,1)=s(5); TSK(c,1)=s(6)
    XLAND(c,1)=s(7); WSPD(c,1)=s(8); BR(c,1)=s(9); PSIM(c,1)=s(10); PSIH(c,1)=s(11); GZ1OZ0(c,1)=s(12)
    HT(c,1)=s(13); LH(c,1)=s(14); SNOW(c,1)=s(15); SICE(c,1)=s(16); AKHS(c,1)=s(17); AKMS(c,1)=s(18)
    THZ0(c,1)=s(19); QZ0(c,1)=s(20); QSFC(c,1)=s(21); CHKLOWQ(c,1)=s(22); CT(c,1)=s(23)
    UZ0(c,1)=s(24); VZ0(c,1)=s(25)
    DO k = 1, kx
      READ(10,*) row   ! U V T TH QV QC QI P PI DZ Z TKE
      U(c,k,1)=row(1); V(c,k,1)=row(2); T(c,k,1)=row(3); TH(c,k,1)=row(4); QV(c,k,1)=row(5)
      QC(c,k,1)=row(6); QI(c,k,1)=row(7); P(c,k,1)=row(8); PII(c,k,1)=row(9); DZ(c,k,1)=row(10)
      ZF(c,k,1)=row(11); TKE(c,k,1)=row(12)
    END DO
    READ(10,*) (PINT(c,k,1), k=1,kx+1)
  END DO
  CLOSE(10)
  LOWLYR=1; KPBL=0; PBLH=0.; MIXHT=0.; EXCH_H=0.; EL=0.
  RUB=0.;RVB=0.;RTHB=0.;RQVB=0.;RQCB=0.;RQIB=0.

  IF (TRIM(fq) == 'T') THEN
    CALL myjpbl(DT=DT,STEPBL=1,HT=HT,DZ=DZ,PMID=P,PINT=PINT,TH=TH,T=T,EXNER=PII,QV=QV,QCW=QC,QCI=QI, &
                U=U,V=V,RHO=RHO,TSK=TSK,QSFC=QSFC,CHKLOWQ=CHKLOWQ,THZ0=THZ0,QZ0=QZ0,UZ0=UZ0,VZ0=VZ0, &
                LOWLYR=LOWLYR,XLAND=XLAND,SICE=SICE,SNOW=SNOW,TKE_MYJ=TKE,EXCH_H=EXCH_H,USTAR=UST,ZNT=ZNT, &
                EL_MYJ=EL,PBLH=PBLH,KPBL=KPBL,CT=CT,AKHS=AKHS,AKMS=AKMS,ELFLX=LH,MIXHT=MIXHT, &
                RUBLTEN=RUB,RVBLTEN=RVB,RTHBLTEN=RTHB,RQVBLTEN=RQVB,RQCBLTEN=RQCB,RQIBLTEN=RQIB, &
                IDS=ids,IDE=ide,JDS=jds,JDE=jde,KDS=kds,KDE=kde,IMS=ims,IME=ime,JMS=jms,JME=jme, &
                KMS=kms,KME=kme,ITS=its,ITE=ite,JTS=jts,JTE=jte,KTS=kts,KTE=kte)
  ELSE
    CALL myjpbl(DT=DT,STEPBL=1,HT=HT,DZ=DZ,PMID=P,PINT=PINT,TH=TH,T=T,EXNER=PII,QV=QV,QCW=QC, &
                U=U,V=V,RHO=RHO,TSK=TSK,QSFC=QSFC,CHKLOWQ=CHKLOWQ,THZ0=THZ0,QZ0=QZ0,UZ0=UZ0,VZ0=VZ0, &
                LOWLYR=LOWLYR,XLAND=XLAND,SICE=SICE,SNOW=SNOW,TKE_MYJ=TKE,EXCH_H=EXCH_H,USTAR=UST,ZNT=ZNT, &
                EL_MYJ=EL,PBLH=PBLH,KPBL=KPBL,CT=CT,AKHS=AKHS,AKMS=AKMS,ELFLX=LH,MIXHT=MIXHT, &
                RUBLTEN=RUB,RVBLTEN=RVB,RTHBLTEN=RTHB,RQVBLTEN=RQVB,RQCBLTEN=RQCB, &
                IDS=ids,IDE=ide,JDS=jds,JDE=jde,KDS=kds,KDE=kde,IMS=ims,IME=ime,JMS=jms,JME=jme, &
                KMS=kms,KME=kme,ITS=its,ITE=ite,JTS=jts,JTE=jte,KTS=kts,KTE=kte)
  END IF

  WRITE(*,'(A,I0,1X,I0)') 'NCOL_KX ', ncol, kx
  DO c = 1, ncol
    WRITE(*,'(A,I0,1X,ES26.16E3,1X,I0)') 'PBL ', c, PBLH(c,1), KPBL(c,1)
    DO k = 1, kx
      WRITE(*,'(A,I0,1X,I0,6(1X,ES26.16E3))') 'TEN ', c, k, RUB(c,k,1), RVB(c,k,1), RTHB(c,k,1), &
            RQVB(c,k,1), RQCB(c,k,1), RQIB(c,k,1)
    END DO
  END DO
END PROGRAM myj_moist_oracle
