"""Stage the real WRF MFSHCONVPBL -> QNSEPBL caller and audit determinism.

Adds diagnostic dumps to OUR driver, never edits WRF physics sources. The
standalone mfshconv=0 oracle and its frozen driver hash remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

from build_oracle import SOURCES

MF_FIELDS = ('massflux_EDKF','entr_EDKF','detr_EDKF','thl_up','thv_up','rt_up',
             'rv_up','rc_up','u_up','v_up','frac_up','rc_mf')


def driver_text():
    text = Path(__file__).with_name('oracle.f90').read_text()
    text = text.replace('  use module_model_constants,',
                        '  use module_bl_mfshconvpbl, only: mfshconvpbl\n  use module_model_constants,',1)
    declarations = '  real, dimension(1,n+1,1) :: '+','.join(MF_FIELDS)+'\n'
    text = text.replace('  integer, dimension(1,1) :: lowlyr,ivg,kpbl\n',
                        declarations+'  integer, dimension(1,1) :: lowlyr,ivg,kpbl\n',1)
    call = '''
  rc_mf=0.
  write(*,'(a)') 'INPUT_MF'
  call dump()
  call col('rho',rho)
  call scalar('qfx',qfx(1,1))
  call col('rc_mf',rc_mf)
  call mfshconvpbl(DT=dt,STEPBL=stepbl,HT=ht,DZ=dz, &
      RHO=rho,PMID=p,PINT=pint,TH=th,EXNER=exner,QV=qv,QC=qc,U=u,V=v, &
      HFX=hfx,QFX=qfx,TKE=tke, &
      RUBLTEN=ru,RVBLTEN=rv,RTHBLTEN=rth,RQVBLTEN=rqv,RQCBLTEN=rqc, &
      IDS=1,IDE=2,JDS=1,JDE=2,KDS=1,KDE=n+1, &
      IMS=1,IME=1,JMS=1,JME=1,KMS=1,KME=n+1, &
      ITS=1,ITE=1,JTS=1,JTE=1,KTS=1,KTE=n,KRR=2, &
      MASSFLUX_EDKF=massflux_EDKF,ENTR_EDKF=entr_EDKF,DETR_EDKF=detr_EDKF, &
      THL_UP=thl_up,THV_UP=thv_up,RT_UP=rt_up,RV_UP=rv_up,RC_UP=rc_up, &
      U_UP=u_up,V_UP=v_up,FRAC_UP=frac_up,RC_MF=rc_mf,WTHV=wmf,PLM_BL89=lmf)
  write(*,'(a)') 'OUTPUT_MF'
  call dump()
'''
    for field in MF_FIELDS:
        call += f"  call col('{field}',{field})\n"
    call += "  call col('WTHV',wmf)\n  call col('PLM_BL89',lmf)\n"
    anchor = '  call qnsepbl(dt,stepbl,ht,dz,p,pint,th,t,exner,qv,qc,u,v,rho, &'
    assert text.count(anchor) == 1
    return text.replace(anchor,call+anchor)


def parse(text):
    sections={key:{} for key in ('INPUT_SFC','OUTPUT_SFC','INPUT','OUTPUT','INPUT_MF','OUTPUT_MF')}
    section=None
    for line in text.splitlines():
        if line in sections:
            section=sections[line]
        elif section is not None and '=' in line:
            key,value=line.split('=',1)
            if '[' in key:
                section.setdefault(key.split('[')[0],[]).append(float(value))
            else:
                section[key]=float(value)
    return sections


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wrf',type=Path,required=True)
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fc',default='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    ap.add_argument('--init-logical',choices=('default','false','true'),default='default')
    ap.add_argument('--optimization',choices=('O0','O1','O2','O3'),default='O0')
    ap.add_argument('--compile-timeout',type=int,default=90)
    ap.add_argument('--contract',choices=('off','fast'),default='off')
    ap.add_argument('--native',action='store_true')
    ap.add_argument('--no-vectorize',action='store_true')
    ap.add_argument('--mode',choices=('both','fp32','fp64'),default='both')
    ap.add_argument('--repetitions',type=int,choices=(1,4),default=4)
    args=ap.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit('QUIET')
    args.build.mkdir(parents=True,exist_ok=True)
    hashes={}
    sources=(*SOURCES,'phys/module_bl_mfshconvpbl.F')
    for relative in sources:
        src=args.wrf/relative
        hashes[relative]=hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copyfile(src,args.build/src.name)
    driver=driver_text()
    (args.build/'oracle.f90').write_text(driver)
    (args.build/'module_dm.f90').write_text('module module_dm\ncontains\n'
        'subroutine wrf_dm_maxval(a,b,c)\ninteger :: a,b,c\nstop 98\nend subroutine\nend module\n')
    flags=['-'+args.optimization,'-ffree-form','-ffree-line-length-none','-cpp','-DNMM_CORE=0',
           '-ffp-contract='+args.contract,'-fcheck=all',
           '-finit-real=zero','-finit-integer=0','-finit-logical=false']
    if args.native:flags.append('-march=native')
    if args.no_vectorize:flags.extend(('-fno-tree-vectorize','-fno-tree-slp-vectorize'))
    if args.init_logical != 'default':
        flags.append('-finit-logical='+args.init_logical)
    all_cases={};opcodes={}
    for mode in (('fp32','fp64') if args.mode=='both' else (args.mode,)):
        extra=[] if mode=='fp32' else ['-fdefault-real-8','-fdefault-double-8']
        subprocess.run([args.fc,*flags,*extra,'module_model_constants.F','module_dm.f90',
                        'module_sf_qnsesfc.F','module_bl_qnsepbl.F','module_bl_mfshconvpbl.F',
                        'oracle.f90','-o','oracle'],cwd=args.build,check=True,timeout=args.compile_timeout)
        disassembly=subprocess.check_output(['objdump','-d',str(args.build/'oracle')],text=True,timeout=20)
        fmas=re.findall(r'\b(vf(?:n?madd|n?msub)\w*)\b',disassembly)
        opcodes[mode]={name:fmas.count(name) for name in sorted(set(fmas))}
        all_cases[mode]=[]
        for cid in range(1,16):
            repetitions=[]
            for perturb in (0,85,170,255)[:args.repetitions]:
                env={**os.environ,'MALLOC_PERTURB_':str(perturb)}
                result=subprocess.run([str(args.build/'oracle'),str(cid)],capture_output=True,
                                      text=True,check=True,timeout=20,env=env)
                (args.build.parent/f'{mode}_{cid}_{perturb}.txt').write_text(result.stdout)
                repetitions.append(parse(result.stdout))
            all_cases[mode].append(repetitions)
    packet=dict(scope='Pristine ARW QNSESFC -> MFSHCONVPBL(KRR2,OMIXUVfalse,PIMPL0) -> QNSEPBL',
                source_sha256=hashes,driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
                flags=flags,init_logical=args.init_logical,
                heap_perturbations=[0,85,170,255][:args.repetitions],
                fma_opcodes=opcodes,cases=all_cases)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    print('Wrote 15 cases x 2 precisions x',args.repetitions,'heap perturbations:',args.output)


if __name__=='__main__':
    main()
