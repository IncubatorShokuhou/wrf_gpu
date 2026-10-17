"""WRITE-only old-KF component checkpoints from actual active KFCPS soundings.

Require complete original stdout BYTE after stripping checkpoint lines. All
observations precede/after the pristine subprogram body; no numerical edits.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

reference=Path(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir(parents=True,exist_ok=True)
tmp=work/'tmp';tmp.mkdir(exist_ok=True);env=dict(os.environ,TMPDIR=str(tmp))
report={'source':'pristine WRF module_cu_kf.F actual helper calls','write_only':True,'arms':{}}
specs={
 'DTFRZNEW':('      RV=461.5',
    'TU,P,THTEU,QVAP,QLIQ,QICE,RATIO2,TTFRZ,TBFRZ,QNWFRZ,RL,FRC1,EFFQ,REAL(IFLAG)',
    'TU,THTEU,QVAP,QLIQ,QICE,RATIO2,QNWFRZ,RL,FRC1,REAL(IFLAG)'),
 'TPMIX':('      C5=1.0723E-3', 'P,THTU,TU,QU,QLIQ,QICE,RATIO2,RL', 'TU,QU,QLIQ,QICE,QNEWLQ,QNEWIC'),
 'ENVIRTHT':('      IF(R1.LT.1.E-6)THEN', 'P1,T1,Q1,R1,RL', 'THT1'),
 'CONDLOAD':('      QTOT=QLIQ+QICE', 'QLIQ,QICE,WTW,DZ,BOTERM,ENTERM,RATE,QNEWLQ,QNEWIC',
              'QLIQ,QICE,WTW,QLQOUT,QICOUT,QNEWLQ,QNEWIC'),
 'TPDD':('      ES=ALIQ*EXP((BLIQ*TGS-CLIQ)/(TGS-DLIQ))', 'P,THTED,TGS,RD,RH', 'TPDD,RS'),
}
for mode in ('r4','r8'):
    ref=reference/f'build_{mode}_zero';b=work/mode;b.mkdir(exist_ok=True)
    shutil.copyfile(ref/'module_wrf_error.mod',b/'module_wrf_error.mod')
    shutil.copyfile(ref/'module_wrf_error.o',b/'module_wrf_error.o')
    shutil.copyfile(ref/'kf99_oracle.F90',b/'kf99_oracle.F90')
    original=(ref/'module_cu_kf.F').read_text();source=original
    for name,(anchor,inputs,outputs) in specs.items():
        start=re.search(r'(?im)^\s*(?:SUBROUTINE|FUNCTION)\s+'+name+r'\(',source).start()
        end=re.search(r'(?im)^\s*END\s+(?:SUBROUTINE|FUNCTION)\s+'+name+r'\b',source[start:]).start()+start
        body=source[start:end]
        match=re.search(r'(?m)^'+re.escape(anchor)+r'\s*$',body)
        assert match,name
        line=lambda kind,values:f"      WRITE(*,'(A,*(ES27.17E3))') 'KF99_{name}_{kind} ',{values}\n"
        body=body[:match.start()]+line('IN',inputs)+body[match.start():]+line('OUT',outputs)
        source=source[:start]+body+source[end:]
    stripped='\n'.join(l for l in source.split('\n') if "'KF99_" not in l)
    assert stripped==original
    (b/'module_cu_kf.F').write_text(source)
    flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp',
           '-DEM_CORE=1','-DNMM_CORE=0','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    if mode=='r8':flags+=['-fdefault-real-8','-fdefault-double-8']
    c='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    for args in (flags+['-c','module_cu_kf.F'],flags+['-c','kf99_oracle.F90'],
                 ['-O2','-o','oracle','module_wrf_error.o','module_cu_kf.o','kf99_oracle.o']):
        subprocess.run([c]+args,cwd=b,env=env,check=True)
    records={name:[] for name in specs}
    last={}
    for case in range(1,6):
        raw=subprocess.check_output([str(b/'oracle'),str(case)],cwd=b,env=env)
        kept=[]
        for line in raw.decode().splitlines(keepends=True):
            if not line.startswith('KF99_'):
                kept.append(line);continue
            prefix,*numbers=line.split();_,name,kind=prefix.split('_')
            vals=[float(x) for x in numbers]
            if kind=='IN':last[name]=vals
            else:records[name].append({'case':case,'input':last.pop(name),'output':vals})
        clean=''.join(kept).encode()
        assert clean==(reference/f'case{case}_{mode}_zero.txt').read_bytes(),(mode,case,'observer changed stdout')
    assert not last
    (work/f'helpers_{mode}.json').write_text(json.dumps(records))
    report['arms'][mode]={'stdout_byte':5,'counts':{k:len(v) for k,v in records.items()},
                          'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
                          'write_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'flags':flags}
    (work/'manifest.json').write_text(json.dumps(report,indent=2));print(mode,report['arms'][mode],flush=True)
