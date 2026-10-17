"""WRITE-only KFPARA environment and every buoyant source-layer trigger."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

reference=Path(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir(parents=True,exist_ok=True)
phase=sys.argv[3] if len(sys.argv)>3 else 'trigger'
plume=phase in ('plume','downdraft','closure')
tmp=work/'tmp';tmp.mkdir(exist_ok=True);env=dict(os.environ,TMPDIR=str(tmp))
manifest={'write_only':True,'arms':{}}
for mode in ('r4','r8'):
    ref=reference/f'build_{mode}_zero';b=work/mode;b.mkdir(exist_ok=True)
    for suffix in ('.mod','.o'):
        shutil.copyfile(ref/('module_wrf_error'+suffix),b/('module_wrf_error'+suffix))
    shutil.copyfile(ref/'kf99_oracle.F90',b/'kf99_oracle.F90')
    original=(ref/'module_cu_kf.F').read_text();source=original
    pattern=r'^        DZA\(KL\)=0\.\s*$'
    writes=''
    for name in ('Q0','QES','TV0','DP','Z0','DZA'):
        writes+=f"\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_ENV_{name} ',{name}"
    writes+="\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_ENV_LEVELS ',REAL(ML),REAL(L5),REAL(L4),REAL(LLFC)"
    source,n=re.subn(pattern,lambda m:m.group(0)+writes,source,flags=re.M);assert n==1
    values='REAL(LC),REAL(KPBL),REAL(K),REAL(KLCL),ZLCL,TLCL,TVLCL,TVEN,TMIX,QMIX,PMIX,ZMIX,DPTHMX,WLCL,THETEU(K),THTES(K),VMFLCL,AU0'
    pattern=r'^        UPNEW=UPOLD\s*$'
    source,n=re.subn(pattern,lambda m:m.group(0)+f"\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_TRIGGER ',{values}",source,flags=re.M);assert n==1
    if plume:
        pattern=r'^        CLDHGT=Z0\(LTOP\)-ZLCL\s*$'
        values='REAL(LC),REAL(LET),REAL(LTOP),REAL(KFRZ),CLDHGT,ABE,UPOLD,UPNEW,WTW'
        source,n=re.subn(pattern,lambda m:m.group(0)+f"\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_PLUME_SCALARS ',{values}",source,flags=re.M);assert n==1
        pattern=r'^[ \t]*85[ \t]+CONTINUE[ \t]*$'
        writes="\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_PLUME_LC ',REAL(LC)"
        for name in ('TU','TVU','QU','WU','UMF','UER','UDR','QLIQ','QICE','QLQOUT','QICOUT',
                     'DETLQ','DETIC','PPTLIQ','PPTICE','RATIO2','THETEU','THETEE','THTES','QDT','EQFRC'):
            writes+=f"\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_PLUME_{name} ',{name}(1:LTOP)"
        writes+="\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_PLUME_TRPPT ',TRPPT"
        source,n=re.subn(pattern,lambda m:m.group(0)+writes,source,flags=re.M);assert n==1
    if phase in ('downdraft','closure'):
        pattern=r'^[ \t]*165[ \t]+AINCMX=1000\.[ \t]*$'
        values='REAL(LC),REAL(LFS),REAL(LDB),REAL(LDT),REAL(LVF),TADVEC,TIMEC,REAL(NIC),USR,PEFF,PEFF2,TDER,PPTFLX,CPR,UPDINC,CNDTNF'
        writes=f"\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_DD_SCALARS ',{values}"
        for name in ('UMF','UER','UDR','DETLQ','DETIC','PPTLIQ','PPTICE','DMF','DER','DDR','TZ','QD',
                     'THTAD','THTAU','THTA0','EMS','EMSD'):
            writes+=f"\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_DD_{name} ',{name}(1:LTOP)"
        writes+="\n        WRITE(*,'(A,*(ES27.17E3))') 'KF99_DD_THETED ',THETED(LDB:LFS)"
        source,n=re.subn(pattern,lambda m:m.group(0)+writes,source,flags=re.M);assert n==1
    if phase=='closure':
        pattern=r'^[ \t]*IF\(NOITR\.EQ\.1\)THEN[ \t]*$'
        values='REAL(NCOUNT),REAL(NOITR),AINC,AINCMX,ABE,ABEG,TIMEC,DTT,REAL(NSTEP),DTIME,PPTFLX,TDER,REAL(K),REAL(KLCL),ZLCL,TLCL,PLCL,THMIX,TMIX,QMIX,PMIX'
        writes=f"        WRITE(*,'(A,*(ES27.17E3))') 'KF99_CLOSURE_SCALARS ',{values}\n"
        for name in ('OMG','FXM','TG','QG','THTAG','THPA','QPA'):
            writes+=f"        WRITE(*,'(A,*(ES27.17E3))') 'KF99_CLOSURE_{name} ',{name}(1:LTOP)\n"
        source,n=re.subn(pattern,lambda m:writes+m.group(0),source,flags=re.M);assert n==1
    assert '\n'.join(l for l in source.split('\n') if "'KF99_" not in l)==original
    (b/'module_cu_kf.F').write_text(source)
    flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp',
           '-DEM_CORE=1','-DNMM_CORE=0','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    if mode=='r8':flags+=['-fdefault-real-8','-fdefault-double-8']
    c='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    for args in (flags+['-c','module_cu_kf.F'],flags+['-c','kf99_oracle.F90'],
                 ['-O2','-o','oracle','module_wrf_error.o','module_cu_kf.o','kf99_oracle.o']):
        subprocess.run([c]+args,cwd=b,env=env,check=True)
    records={}
    for case in range(1,6):
        raw=subprocess.check_output([str(b/'oracle'),str(case)],cwd=b,env=env)
        kept=[];rec={'environment':{},'triggers':[],'plume_scalars':[],'plumes':[],'downdraft':{},'closure':[]}
        for line in raw.decode().splitlines(keepends=True):
            if not line.startswith('KF99_'):kept.append(line);continue
            prefix,*vals=line.split();numbers=[float(x) for x in vals]
            if prefix=='KF99_TRIGGER':rec['triggers'].append(numbers)
            elif prefix=='KF99_PLUME_SCALARS':rec['plume_scalars'].append(numbers)
            elif prefix=='KF99_PLUME_LC':rec['plumes'].append({'LC':int(numbers[0]),'profiles':{}})
            elif prefix.startswith('KF99_PLUME_'):rec['plumes'][-1]['profiles'][prefix.removeprefix('KF99_PLUME_')]=numbers
            elif prefix.startswith('KF99_DD_'):rec['downdraft'][prefix.removeprefix('KF99_DD_')]=numbers
            elif prefix=='KF99_CLOSURE_SCALARS':rec['closure'].append({'scalars':numbers,'profiles':{}})
            elif prefix.startswith('KF99_CLOSURE_'):rec['closure'][-1]['profiles'][prefix.removeprefix('KF99_CLOSURE_')]=numbers
            else:rec['environment'][prefix.removeprefix('KF99_ENV_')]=numbers
        assert ''.join(kept).encode()==(reference/f'case{case}_{mode}_zero.txt').read_bytes(),(mode,case)
        records[str(case)]=rec
    (work/f'trigger_{mode}.json').write_text(json.dumps(records))
    manifest['arms'][mode]={'stdout_byte':5,'trigger_counts':{k:len(v['triggers']) for k,v in records.items()},
        'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'write_source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2));print(mode,manifest['arms'][mode],flush=True)
