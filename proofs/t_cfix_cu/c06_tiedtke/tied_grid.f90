! Whole-patch driver for the UNMODIFIED WRF module_cu_tiedtke.F (c06 real-data oracle, t-cfix-cu).
! Reads float32 stream 'in.bin' (Fortran (i,k,j) order), calls CU_TIEDTKE exactly as
! cumulus_driver.F:1394-1412 does, writes 'out.bin'.
PROGRAM tied_grid
  USE module_cu_tiedtke, ONLY : cu_tiedtke
  IMPLICIT NONE
  INTEGER :: nx, ny, nz, itimestep, stepcu
  REAL :: dt
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: U,V,W,T,QV,QC,QI,PII,RHO,QVFTEN,QVPBLTEN,DZ8W,PCPS,P8W
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN,RUCUTEN,RVCUTEN
  REAL, ALLOCATABLE, DIMENSION(:,:) :: RAINCV, PRATEC, QFX, XLAND
  LOGICAL, ALLOCATABLE, DIMENSION(:,:) :: CU_ACT_FLAG
  REAL, ALLOCATABLE :: ZNU(:)
  INTEGER :: ids,ide,jds,jde,kds,kde
  OPEN(10, FILE='in.bin', ACCESS='stream', FORM='unformatted', STATUS='old')
  READ(10) nx, ny, nz, itimestep, stepcu, dt
  ! WRF memory: i 1..nx+1 (ide = nx+1), k 1..nz+1 (kde = nz+1), j 1..ny+1; tiles its..ite = 1..nx
  ids=1; ide=nx+1; jds=1; jde=ny+1; kds=1; kde=nz+1
  ALLOCATE(U(nx,nz+1,ny),V(nx,nz+1,ny),W(nx,nz+1,ny),T(nx,nz+1,ny),QV(nx,nz+1,ny),QC(nx,nz+1,ny),QI(nx,nz+1,ny))
  ALLOCATE(PII(nx,nz+1,ny),RHO(nx,nz+1,ny),QVFTEN(nx,nz+1,ny),QVPBLTEN(nx,nz+1,ny),DZ8W(nx,nz+1,ny))
  ALLOCATE(PCPS(nx,nz+1,ny),P8W(nx,nz+1,ny))
  ALLOCATE(RTHCUTEN(nx,nz+1,ny),RQVCUTEN(nx,nz+1,ny),RQCCUTEN(nx,nz+1,ny),RQICUTEN(nx,nz+1,ny))
  ALLOCATE(RUCUTEN(nx,nz+1,ny),RVCUTEN(nx,nz+1,ny))
  ALLOCATE(RAINCV(nx,ny),PRATEC(nx,ny),QFX(nx,ny),XLAND(nx,ny),CU_ACT_FLAG(nx,ny),ZNU(nz+1))
  U=0.; V=0.; W=0.; T=0.; QV=0.; QC=0.; QI=0.; PII=0.; RHO=0.; QVFTEN=0.; QVPBLTEN=0.; DZ8W=0.; PCPS=0.; P8W=0.
  READ(10) U, V, W, T, QV, QC, QI, PII, RHO, QVFTEN, QVPBLTEN, DZ8W, PCPS, P8W, QFX, XLAND, ZNU
  CLOSE(10)
  RTHCUTEN=0.; RQVCUTEN=0.; RQCCUTEN=0.; RQICUTEN=0.; RUCUTEN=0.; RVCUTEN=0.; RAINCV=0.; PRATEC=0.
  CALL CU_TIEDTKE(DT=dt,ITIMESTEP=itimestep,STEPCU=stepcu,RAINCV=RAINCV,PRATEC=PRATEC,QFX=QFX,ZNU=ZNU, &
       U3D=U,V3D=V,W=W,T3D=T,PI3D=PII,RHO3D=RHO,QV3D=QV,QC3D=QC,QI3D=QI,QVPBLTEN=QVPBLTEN, &
       DZ8W=DZ8W,PCPS=PCPS,P8W=P8W,XLAND=XLAND,CU_ACT_FLAG=CU_ACT_FLAG, &
       IDS=ids,IDE=ide,JDS=jds,JDE=jde,KDS=kds,KDE=kde, &
       IMS=1,IME=nx,JMS=1,JME=ny,KMS=1,KME=nz+1, &
       ITS=1,ITE=nx,JTS=1,JTE=ny,KTS=1,KTE=nz, &
       QVFTEN=QVFTEN,RTHCUTEN=RTHCUTEN,RQVCUTEN=RQVCUTEN,RQCCUTEN=RQCCUTEN,RQICUTEN=RQICUTEN, &
       RUCUTEN=RUCUTEN,RVCUTEN=RVCUTEN,F_QV=.TRUE.,F_QC=.TRUE.,F_QR=.TRUE.,F_QI=.TRUE.,F_QS=.TRUE.)
  OPEN(11, FILE='out.bin', ACCESS='stream', FORM='unformatted', STATUS='replace')
  WRITE(11) RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN, RUCUTEN, RVCUTEN, RAINCV, PRATEC
  CLOSE(11)
END PROGRAM tied_grid
