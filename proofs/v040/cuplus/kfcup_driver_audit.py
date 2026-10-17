"""Full original KF-CuP CPS oracle, chem/aerosol feedback disabled.

Run CPU-only with the lane affinity, QUIET guard and timeout. Fresh private
copies of the complete original KF-CuP and mixactivate modules are compiled;
their otherwise-unused radiation dependency is an exact extracted original
cal_cldfra2 procedure, not a numerical stub. Only the state-description ABI
is reused from the pristine build. No full WRF executable is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from generate import extract
from mskf_driver_audit import replace_once


ROOT=Path(__file__).resolve().parents[3]


def driver_source(parcel=False):
    text=(ROOT/'proofs/p0_4/oracle/kf_oracle_driver.f90').read_text()
    text=replace_once(text,'USE module_cu_kfeta, ONLY : kf_eta_cps, kf_eta_init',
                      'USE module_cu_kfcup, ONLY : KF_cup_CPS, KF_cup_init')
    text=replace_once(text,'ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX',
                      'ims=1, ime=5, jms=1, jme=5, kms=1, kme=KX+1')
    text=replace_once(text,'ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1',
                      'ids=1, ide=6, jds=1, jde=6, kds=1, kde=KX+1')
    # One active column with real surrounding-array bounds avoids WRF's
    # single-column diagnostic branch without duplicating expensive ensembles.
    text=replace_once(text,'its=1, ite=1, jts=1, jte=1, kts=1, kte=KX',
                      'its=3, ite=3, jts=3, jte=3, kts=1, kte=KX')
    text=replace_once(text,'INTEGER :: k, case_id','INTEGER :: k, case_id, ii, jj')
    text=replace_once(text,'REAL, DIMENSION(KX) :: zlev','''REAL, DIMENSION(KX) :: zlev
  REAL :: BACKGROUND_WIND, SURFACE_HEAT, PBL_DEPTH
  INTEGER, PARAMETER :: NUMBINS=5
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: Z,ZW,CLDFRA_CUP,CLDFRATEND_CUP,LNTERMS
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: WUP_CUP,QC_IC_CUP,QNDROP_IC_CUP,QC_IU_CUP
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: FCVT_QC_TO_PR_CUP,FCVT_QC_TO_QI_CUP,FCVT_QI_TO_PR_CUP
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: MFUP_CUP,MFUP_ENT_CUP,MFDN_CUP,MFDN_ENT_CUP,UPDFRA_CUP
  REAL, DIMENSION(ims:ime,jms:jme) :: XLAND,PSFC,HT,TSK,HFX,QFX,MAVAIL,BR,REGIME,PBLH,T2,Q2
  REAL, DIMENSION(ims:ime,jms:jme) :: SLOPESFC,SLOPEEZ,SIGMASFC,SIGMAEZ,TAUCLOUD,TACTIVE,ACTIVEFRAC,TSTAR,LNINT
  REAL, DIMENSION(ims:ime,jms:jme) :: WCLOUDBASE,WACT_CUP,WULCL_CUP,TCLOUD_CUP
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL
  LOGICAL, DIMENSION(ims:ime,jms:jme) :: CUPFLAG
''')
    text=replace_once(text,'  ! --- model timestep / grid', '''
  BACKGROUND_WIND=2.;SURFACE_HEAT=150.;PBL_DEPTH=1500.
  IF(COMMAND_ARGUMENT_COUNT() >= 2) THEN
    CALL GET_COMMAND_ARGUMENT(2,arg);READ(arg,*)BACKGROUND_WIND
  ENDIF
  IF(COMMAND_ARGUMENT_COUNT() >= 3) THEN
    CALL GET_COMMAND_ARGUMENT(3,arg);READ(arg,*)SURFACE_HEAT
  ENDIF
  IF(COMMAND_ARGUMENT_COUNT() >= 4) THEN
    CALL GET_COMMAND_ARGUMENT(4,arg);READ(arg,*)PBL_DEPTH
  ENDIF
  ! --- model timestep / grid''')
    start=text.index('  CALL kf_eta_init(')
    end=text.index('  ! --- build the chosen sounding',start)
    text=text[:start]+'''  CALL KF_cup_init(RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN,RQICUTEN,RQSCUTEN, &
       NCA,W0AVG,2,3,SVP1,SVP2,SVP3,SVPT0,CUPFLAG,CLDFRA_CUP,CLDFRATEND_CUP, &
       SHALL,TCLOUD_CUP,1,.FALSE.,.TRUE., &
       ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)

'''+text[end:]
    start=text.index('  CALL KF_ETA_CPS(')
    end=text.index('  ! ---------------- dump everything',start)
    setup='''
  DO jj=jms,jme
    DO ii=ims,ime
      DO k=kts,kte
        T(ii,k,jj)=T(1,k,1);QV(ii,k,jj)=QV(1,k,1)
        PCPS(ii,k,jj)=PCPS(1,k,1);RHO(ii,k,jj)=RHO(1,k,1)
        DZ8W(ii,k,jj)=DZ8W(1,k,1);U(ii,k,jj)=U(1,k,1);V(ii,k,jj)=V(1,k,1)
        PII(ii,k,jj)=PII(1,k,1);TH(ii,k,jj)=TH(1,k,1)
        W(ii,k,jj)=W(1,k,1);W0AVG(ii,k,jj)=W0AVG(1,k,1)
        Z(ii,k,jj)=zlev(k)
      ENDDO
      ZW(ii,1,jj)=0.
      DO k=kts,kte
        ZW(ii,k+1,jj)=ZW(ii,k,jj)+DZ8W(ii,k,jj)
      ENDDO
      W(ii,KX+1,jj)=0.
    ENDDO
  ENDDO
  U(:,kts:kte,:)=U(:,kts:kte,:)+BACKGROUND_WIND
  RTHCUTEN=0.;RQVCUTEN=0.;RQCCUTEN=0.;RQRCUTEN=0.;RQICUTEN=0.;RQSCUTEN=0.;NCA=-100.
  CLDFRA_CUP=0.;CLDFRATEND_CUP=0.;CUPFLAG=.FALSE.;SHALL=2.
  WUP_CUP=0.;QC_IC_CUP=0.;QNDROP_IC_CUP=0.;QC_IU_CUP=0.
  FCVT_QC_TO_PR_CUP=0.;FCVT_QC_TO_QI_CUP=0.;FCVT_QI_TO_PR_CUP=0.
  MFUP_CUP=0.;MFUP_ENT_CUP=0.;MFDN_CUP=0.;MFDN_ENT_CUP=0.;UPDFRA_CUP=0.
  WACT_CUP=0.;WULCL_CUP=0.;TCLOUD_CUP=0.
  XLAND=1.;PSFC=100000.;HT=0.;TSK=T(1,1,1);HFX=SURFACE_HEAT;QFX=1.E-4;MAVAIL=1.
  BR=-.1;REGIME=4.;PBLH=PBL_DEPTH;KPBL=6;T2=T(1,1,1);Q2=QV(1,1,1)
  CALL KF_cup_CPS(GRID_ID=1, &
       ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
       ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
       its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte, &
       DT=DT,KTAU=KTAU,DX=DX,RHO=RHO,RAINCV=RAINCV,NCA=NCA, &
       U=U,V=V,TH=TH,T=T,W=W,DZ8W=DZ8W,PCPS=PCPS,PI=PII,W0AVG=W0AVG, &
       XLV0=XLV0,XLV1=XLV1,XLS0=XLS0,XLS1=XLS1,CP=CP,R=R_D,G=G,EP1=EP_1,EP2=EP_2, &
       SVP1=SVP1,SVP2=SVP2,SVP3=SVP3,SVPT0=SVPT0,STEPCU=STEPCU,CU_ACT_FLAG=CU_ACT_FLAG, &
       warm_rain=warm_rain,CUTOP=CUTOP,CUBOT=CUBOT,QV=QV,XLAND=XLAND, &
       PSFC=PSFC,Z=Z,Z_AT_W=ZW,HT=HT,TSK=TSK,HFX=HFX,QFX=QFX,MAVAIL=MAVAIL,SF_SFCLAY_PHYSICS=1, &
       BR=BR,REGIME=REGIME,PBLH=PBLH,KPBL=KPBL,T2=T2,Q2=Q2, &
       SLOPESFC=SLOPESFC,SLOPEEZ=SLOPEEZ,SIGMASFC=SIGMASFC,SIGMAEZ=SIGMAEZ, &
       CUPFLAG=CUPFLAG,CLDFRA_CUP=CLDFRA_CUP,CLDFRATEND_CUP=CLDFRATEND_CUP,SHALL=SHALL, &
       TAUCLOUD=TAUCLOUD,TACTIVE=TACTIVE,ACTIVEFRAC=ACTIVEFRAC,TSTAR=TSTAR,LNTERMS=LNTERMS,LNINT=LNINT, &
       NUMBINS=NUMBINS,THBINSIZE=.1,RBINSIZE=1.E-4,MINDEEPFREQ=.1,MINSHALLOWFREQ=.001, &
       WCLOUDBASE=WCLOUDBASE,WACT_CUP=WACT_CUP,WULCL_CUP=WULCL_CUP,WUP_CUP=WUP_CUP, &
       QC_IC_CUP=QC_IC_CUP,QNDROP_IC_CUP=QNDROP_IC_CUP,QC_IU_CUP=QC_IU_CUP, &
       FCVT_QC_TO_PR_CUP=FCVT_QC_TO_PR_CUP,FCVT_QC_TO_QI_CUP=FCVT_QC_TO_QI_CUP,FCVT_QI_TO_PR_CUP=FCVT_QI_TO_PR_CUP, &
       MFUP_CUP=MFUP_CUP,MFUP_ENT_CUP=MFUP_ENT_CUP,MFDN_CUP=MFDN_CUP,MFDN_ENT_CUP=MFDN_ENT_CUP, &
       UPDFRA_CUP=UPDFRA_CUP,TCLOUD_CUP=TCLOUD_CUP,SHCU_AEROSOLS_OPT=0, &
       F_QV=F_QV,F_QC=F_QC,F_QR=F_QR,F_QI=F_QI,F_QS=F_QS, &
       RTHCUTEN=RTHCUTEN,RQVCUTEN=RQVCUTEN,RQCCUTEN=RQCCUTEN,RQRCUTEN=RQRCUTEN,RQICUTEN=RQICUTEN,RQSCUTEN=RQSCUTEN)
  PRATEC=RAINCV/DT
'''
    text=text[:start]+setup+text[end:]
    text=text.replace('arr(1,kk,1)','arr(3,kk,3)')
    for name in ('RAINCV','PRATEC','NCA','CUTOP','CUBOT','SHALL'):
        text=''.join(l.replace(name+'(1,1)',name+'(3,3)') if l.lstrip().upper().startswith('WRITE') else l
                     for l in text.splitlines(keepends=True))
    text=''.join(l for l in text.splitlines(keepends=True) if "'TIMEC='" not in l)
    extra='''
  WRITE(*,'(A,I0)') 'CUPFLAG=',MERGE(1,0,CUPFLAG(3,3))
  IF(CUPFLAG(3,3)) WRITE(*,'(A,ES23.15)') 'ACTIVEFRAC=',ACTIVEFRAC(3,3)
  CALL dump_col('CLDFRA_CUP',CLDFRA_CUP)
  CALL dump_col('QC_IC_CUP',QC_IC_CUP)
  CALL dump_col('MFUP_CUP',MFUP_CUP)
  CALL dump_col('MFDN_CUP',MFDN_CUP)
  CALL dump_col('UPDFRA_CUP',UPDFRA_CUP)
  CALL dump_col('Z',Z)
  CALL dump_col('ZW',ZW)
  CALL dump_col('TH',TH)
  CALL dump_col('PI',PII)
  IF(CUPFLAG(3,3).AND.NINT(SHALL(3,3))==1) THEN
    WRITE(*,'(A,ES23.15)') 'TAUCLOUD=',TAUCLOUD(3,3)
    WRITE(*,'(A,ES23.15)') 'TACTIVE=',TACTIVE(3,3)
    WRITE(*,'(A,ES23.15)') 'LNINT=',LNINT(3,3)
  ENDIF
'''
    text=replace_once(text,'\nCONTAINS\n',extra+'\nCONTAINS\n')
    if not parcel:
        text=replace_once(text,'  CALL KF_cup_CPS(GRID_ID=1,',
                          "  CALL dump_col('IN_W0AVG',W0AVG)\n  CALL dump_col('IN_W',W)\n  CALL KF_cup_CPS(GRID_ID=1,")
    if parcel:
        text=parcel_driver(text)
    return text


def parcel_driver(text):
    """Expose the original PARA boundary separately from the CPS ensemble."""
    text=text.replace('ONLY : KF_cup_CPS, KF_cup_init', 'ONLY : KF_cup_CPS, KF_cup_init, KF_cup_PARA')
    text=replace_once(text,'  REAL :: BACKGROUND_WIND, SURFACE_HEAT, PBL_DEPTH', '''  REAL :: BACKGROUND_WIND, SURFACE_HEAT, PBL_DEPTH
  REAL :: PERT_T,PERT_Q,PARCEL_FREQ,WLCL,UPAREA,WULCL,WACT
  INTEGER :: PARCEL_FLAG,ISHALL
  REAL,DIMENSION(KX) :: QLG,QIG,WUP,QCLIQ,QCICE,QN,UMF,UER,UDR,DMF,DER,DDR,FQCQI,FQCPR,FQIPR
  REAL,DIMENSION(KX,1) :: CHEM1D
  REAL,DIMENSION(KX) :: DQDT,DQIDT,DQCDT,DQRDT,DQSDT,DTDT''')
    start=text.index('  CALL KF_cup_CPS(GRID_ID=1,')
    end=text.index('  PRATEC=RAINCV/DT',start)
    text=text[:start]+'''  PERT_T=0.;PERT_Q=0.;PARCEL_FREQ=.04;PARCEL_FLAG=1
  IF(COMMAND_ARGUMENT_COUNT() >= 5) THEN
    CALL GET_COMMAND_ARGUMENT(5,arg);READ(arg,*)PERT_T
  ENDIF
  IF(COMMAND_ARGUMENT_COUNT() >= 6) THEN
    CALL GET_COMMAND_ARGUMENT(6,arg);READ(arg,*)PERT_Q
  ENDIF
  IF(COMMAND_ARGUMENT_COUNT() >= 7) THEN
    CALL GET_COMMAND_ARGUMENT(7,arg);READ(arg,*)PARCEL_FREQ
  ENDIF
  IF(COMMAND_ARGUMENT_COUNT() >= 8) THEN
    CALL GET_COMMAND_ARGUMENT(8,arg);READ(arg,*)PARCEL_FLAG
  ENDIF
  CUPFLAG=PARCEL_FLAG/=0
  CUTOP=1.;CUBOT=REAL(KX+1)
  DQDT=0.;DQIDT=0.;DQCDT=0.;DQRDT=0.;DQSDT=0.;DTDT=0.
  QLG=0.;QIG=0.;CHEM1D=0.
  CALL KF_cup_PARA(GRID_ID=1,KTAU=KTAU,I=3,J=3, &
       U0=U(3,kts:kte,3),V0=V(3,kts:kte,3),T0=T(3,kts:kte,3),QV0=QV(3,kts:kte,3), &
       P0=PCPS(3,kts:kte,3),DZQ=DZ8W(3,kts:kte,3),RHOE=RHO(3,kts:kte,3),W0AVG1D=W0AVG(3,kts:kte,3), &
       DT=DT,DX=DX,DXSQ=DX*DX,XLV0=XLV0,XLV1=XLV1,XLS0=XLS0,XLS1=XLS1,CP=CP,R=R_D,G=G, &
       EP2=EP_2,SVP1=SVP1,SVP2=SVP2,SVP3=SVP3,SVPT0=SVPT0, &
       PBLH=PBL_DEPTH,Z_AT_W1D=ZW(3,kts:kte,3),CUPFLAG=CUPFLAG(3,3),TH_PERTURB=PERT_T,R_PERTURB=PERT_Q, &
       FREQ=PARCEL_FREQ,ISHALL=ISHALL,QLG=QLG,QIG=QIG, &
       DQDT=DQDT,DQIDT=DQIDT,DQCDT=DQCDT,DQRDT=DQRDT,DQSDT=DQSDT,DTDT=DTDT, &
       RAINCV=RAINCV,NCA=NCA,NTST=STEPCU,F_QI=F_QI,F_QS=F_QS,WARM_RAIN=WARM_RAIN, &
       CUTOP=CUTOP,CUBOT=CUBOT,WLCL=WLCL, &
       ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
       ims=ims,ime=ime,jms=jms,jme=jms,kms=kms,kme=kme, &
       its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte, &
       IDIAGEE=0,UPDFRA=UPAREA,WULCL=WULCL,WUP=WUP,UMFOUT=UMF,UEROUT=UER,UDROUT=UDR, &
       DMFOUT=DMF,DEROUT=DER,DDROUT=DDR,SHCU_AEROSOLS_OPT=0,FLAG_CHEM=.FALSE.,NUM_CHEM=1, &
       WACT=WACT,QNDROP1D=QN,QC1D=QCLIQ,QI1D=QCICE, &
       FCVT_QC_TO_QI=FQCQI,FCVT_QC_TO_PR=FQCPR,FCVT_QI_TO_PR=FQIPR,CHEM1D=CHEM1D, &
       MAXD_ACOMP=1,MAXD_APHASE=1,MAXD_ATYPE=1,MAXD_ASIZE=1)
  SHALL(3,3)=REAL(ISHALL)
  RTHCUTEN(3,kts:kte,3)=DTDT/PII(3,kts:kte,3)
  RQVCUTEN(3,kts:kte,3)=DQDT;RQCCUTEN(3,kts:kte,3)=DQCDT;RQRCUTEN(3,kts:kte,3)=DQRDT
  RQICUTEN(3,kts:kte,3)=DQIDT;RQSCUTEN(3,kts:kte,3)=DQSDT
''' + text[end:]
    # Full allocated horizontal bounds must be passed, including jme=5.
    text=text.replace('jms=jms,jme=jms,','jms=jms,jme=jme,')
    extra='''
  WRITE(*,'(A,ES23.15)') 'UPDFRA=',UPAREA
  WRITE(*,'(A,ES23.15)') 'WLCL=',WLCL
  WRITE(*,'(A,ES23.15)') 'WULCL=',WULCL
  WRITE(*,'(A,ES23.15)') 'TH_PERTURB=',PERT_T
  WRITE(*,'(A,ES23.15)') 'R_PERTURB=',PERT_Q
  WRITE(*,'(A,ES23.15)') 'FREQ=',PARCEL_FREQ
'''
    for name,var in [('QLG','QLG'),('QIG','QIG'),('QLIQ','QCLIQ'),('QICE','QCICE'),('WUP','WUP'),
                     ('UMF','UMF'),('UER','UER'),('UDR','UDR'),('DMF','DMF'),('DER','DER'),('DDR','DDR'),
                     ('FCVT_QC_TO_QI','FQCQI'),('FCVT_QC_TO_PR','FQCPR'),('FCVT_QI_TO_PR','FQIPR')]:
        upper='MIN(KX,NINT(CUTOP(3,3)))' if name in ('QLG','QIG') else 'KX'
        extra+=f"  DO k=1,{upper}\n    WRITE(*,'(A,I0,A,ES23.15)') '{name}[',k,']=',{var}(k)\n  ENDDO\n"
    text=replace_once(text,'\nCONTAINS\n',extra+'\nCONTAINS\n')
    # CPS-only diagnostic variables are undefined at the PARA boundary.
    text=''.join(line for line in text.splitlines(keepends=True)
                 if not any(quote in line for quote in ("'ACTIVEFRAC='","'TAUCLOUD='","'TACTIVE='","'LNINT='")))
    return text


def scan_output(stdout):
    arrays={};scalars={}
    for line in stdout.splitlines():
        if '=' not in line:continue
        name,value=line.split('=',1)
        try:number=float(value)
        except ValueError:continue
        if '[' in name:arrays.setdefault(name.split('[')[0],[]).append(number)
        else:scalars[name]=number
    bad={name:sum(not math.isfinite(x) for x in values) for name,values in arrays.items()
         if any(not math.isfinite(x) for x in values)}
    bad.update({name:1 for name,value in scalars.items() if not math.isfinite(value)})
    selected={name:(value if math.isfinite(value) else 'NONFINITE') for name,value in scalars.items()
              if name in ('SHALL','CUPFLAG','RAINCV','NCA','CUTOP','CUBOT','ACTIVEFRAC','TAUCLOUD','TACTIVE','LNINT')}
    return dict(nonfinite=bad,finite=not bad,scalars=selected,
                active_tendency_levels=sum(x!=0 for x in arrays.get('RTHCUTEN',[])),
                active_cloud_levels=sum(x!=0 for x in arrays.get('CLDFRA_CUP',[])))


def trace_parcels(source):
    """PRINT-only local copy; the reference equations are left untouched."""
    source=replace_once(source,'            ksvaa = k  ! rce 11-may-2012', '''            ksvaa = k  ! rce 11-may-2012
            WRITE(0,*) 'TRACE_USL',NU,LC,KCLDLAYER,K,KLCL, &
               TMIX,QMIX,PMIX,ZMIX,TLCL,ZLCL,DTTOT,WLCL,VMFLCL,WKL,THETEU(K)''')
    source=replace_once(source,'              IF(WTW.LT.1.E-3)THEN', '''              WRITE(0,*) 'TRACE_LOAD',NU,NK1,TU(NK1),QU(NK1),QLIQ(NK1),QICE(NK1), &
                 WTW,QLQOUT(NK1),QICOUT(NK1),THETEU(NK1),ENTERM,DZZ,BOTERM
              IF(WTW.LT.1.E-3)THEN''')
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-bank',type=Path)
    parser.add_argument('--sequence',action='store_true')
    parser.add_argument('--parcel',action='store_true')
    parser.add_argument('--trace-source',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=('plain','checked'),default=['plain','checked'])
    parser.add_argument('--physics-native-fma',action='store_true')
    parser.add_argument('--initial-real',choices=('zero','snan'),default='zero',
                        help='standing oracle policy; snan is a separate sensitivity arm, traps remain off')
    args=parser.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    if Path('/tmp/wrf_gpu2_quiet').exists():raise RuntimeError('QUIET')
    build=args.build_dir.resolve();build.mkdir(parents=True,exist_ok=True)
    (build/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(build/'tmp')
    wrf=Path('<USER_HOME>/src/wrf_pristine/WRF')
    inputs={'module_model_constants.F':wrf/'share/module_model_constants.F',
            'module_mixactivate.F':wrf/'phys/module_mixactivate.F',
            'module_cu_kfcup.F':wrf/'phys/module_cu_kfcup.F'}
    for name,path in inputs.items():shutil.copyfile(path,build/name)
    if args.trace_source:
        path=build/'module_cu_kfcup.F'
        path.write_text(trace_parcels(path.read_text()))
    proc,bounds=extract((wrf/'phys/module_radiation_driver.F').read_text(),'cal_cldfra2')
    (build/'radiation_dependency.F90').write_text('module module_radiation_driver\ncontains\n'+proc+'\nend module\n')
    logger=(ROOT/'proofs/p0_4/oracle/module_wrf_error.f90').read_text()
    logger+='''
subroutine wrf_error_fatal(message)
 use module_wrf_error,only: original_fatal=>wrf_error_fatal
 character(len=*),intent(in)::message
 call original_fatal(message)
end subroutine
subroutine wrf_message(message)
 character(len=*),intent(in)::message
 write(*,'(A)')trim(message)
end subroutine
subroutine wrf_error_fatal3(file,line,message)
 character(len=*),intent(in)::file,message
 integer,intent(in)::line
 write(0,'(A,I0,A,A)')file,line,': ',message
 stop 2
end subroutine
subroutine wrf_debug(level,message)
 integer,intent(in)::level
 character(len=*),intent(in)::message
end subroutine
'''
    (build/'logger.f90').write_text(logger)
    bank=None;driver=driver_source(args.parcel)
    if args.profile_bank:
        assert not args.parcel,'real bank qualifies the complete CPS caller'
        from real_column_bank import prepare,bind
        bank=prepare(args.profile_bank,build);driver=bind(driver,bank,'kfcup')
    sequence=None
    if args.sequence:
        assert not args.parcel
        from sequence_oracle import bind as sequence_bind
        driver,sequence=sequence_bind(driver,'kfcup')
    (build/'driver.f90').write_text(driver)
    compiler=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    common=[compiler,'-O2','-g','-fbacktrace','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1',
            '-finit-real='+args.initial_real,'-finit-integer=0','-finit-logical=false',
            '-I'+str(wrf/'frame'),'-I'+str(wrf/'phys')]
    sources=['logger.f90','module_model_constants.F','radiation_dependency.F90','module_mixactivate.F',
             'module_cu_kfcup.F','driver.f90',str(wrf/'frame/module_state_description.o'),'-lmvec','-lm']
    records=[]
    receipt=dict(source_sha256={str(p.relative_to(wrf)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.values()},
                 state_abi_object_sha256=hashlib.sha256((wrf/'frame/module_state_description.o').read_bytes()).hexdigest(),
                 radiation_dependency=dict(lines=bounds,sha256=hashlib.sha256(proc.encode()).hexdigest()),
                 driver_sha256=hashlib.sha256((build/'driver.f90').read_bytes()).hexdigest(),flags=common[1:],records=records,
                 trace_source=args.trace_source, measured_cup_sha256=hashlib.sha256((build/'module_cu_kfcup.F').read_bytes()).hexdigest(),
                 physics_native_fma=args.physics_native_fma,profile_bank=bank,sequence=sequence)
    for mode,extra in [('plain',[]),('checked',['-fcheck=all'])]:
        if mode not in args.modes:continue
        receipt['stage']='COMPILING '+mode;args.output.write_text(json.dumps(receipt,indent=2)+'\n')
        start=time.monotonic()
        if args.physics_native_fma:
            # Freeze the generic input builder and constants. Only the exact
            # original physics source gets the compiler-contraction control;
            # never allow native-FMA sounding generation to change the inputs.
            objects=[]
            for source in sources:
                if source.endswith(('.f90','.F','.F90')):
                    physics_flags=['-march=native','-ffp-contract=fast'] if source=='module_cu_kfcup.F' else []
                    target=Path(source).name+'.o'
                    subprocess.run(common+extra+physics_flags+['-c',source,'-o',target],cwd=build,check=True,timeout=900)
                    objects.append(target)
                else:
                    objects.append(source)
            subprocess.run([compiler,*objects,'-o',mode],cwd=build,check=True,timeout=120)
        else:
            subprocess.run(common+extra+sources+['-o',mode],cwd=build,check=True,timeout=900)
        receipt[mode+'_compile_s']=time.monotonic()-start
        for case in (bank['cases'] if bank else (1,2,3,4)):
            arms=([('cup-base',['150','1500','0','0','.04','1']),
                   ('cup-pert',['150','1500','.1','.0001','.04','1']),
                   ('fallback',['10','1500','0','0','.01','0'])] if args.parcel else
                  [('real',[])] if bank else [(f'hfx{hfx}',[str(hfx),'1500']) for hfx in (150,10)])
            for label,controls in arms:
                result=subprocess.run([str(build/mode),str(case),'2',*controls],cwd=build,capture_output=True,text=True,timeout=120)
                out=build/f'{mode}-case{case}-{label}.stdout';err=out.with_suffix('.stderr')
                out.write_text(result.stdout);err.write_text(result.stderr)
                records.append(dict(mode=mode,case=case,label=label,controls=controls,returncode=result.returncode,stdout=str(out),stderr=str(err),
                                    output=scan_output(result.stdout) if result.returncode==0 else None))
                receipt['stage']='RUNNING '+mode;args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    receipt['stage']='COMPLETE';args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(records,indent=2,allow_nan=False))


if __name__=='__main__':main()
