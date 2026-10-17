"""Original WRF FMA spread with identical original input vectors and tables."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time


def columns(path):
    result={}
    for name,k,value in re.findall(r'^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$',path.read_text(),re.M):
        result.setdefault(name,[]).append(float(value))
    return result


def main(base,fresh,out):
    out.mkdir(parents=True,exist_ok=True)
    compiler='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    common=['-O2','-cpp','-ffree-form','-ffree-line-length-none','-fallow-argument-mismatch','-fconvert=big-endian','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    source=base/'module_mp_thompson.F'
    expected='fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    for name in ('qr_acr_qg_mp38V1.dat','qr_acr_qsV2.dat','freezeH2O.dat','CCN_ACTIVATE.BIN'):
        target=out/name
        if not target.exists(): target.symlink_to(base/name)
    started=time.monotonic()
    subprocess.run([compiler,*common,'-ffp-contract=fast','-march=native',f'-I{base}',f'-J{out}','-c',str(source),'-o',str(out/'thompson_fma.o')],check=True,cwd=out)
    deps=[str(base/(name+'.o')) for name in ('stubs','module_model_constants','libmassv','module_mp_radar')]
    for group,driver in (('base',base/'driver.f90'),('fresh',fresh/'fresh_driver.f90')):
        text=driver.read_text().replace('  CHARACTER(LEN=32) :: arg','  CHARACTER(LEN=32) :: arg\n  CHARACTER(LEN=256) :: frozen_qv_path')
        insertion="""  ! Freeze exact independently recorded caller QV before any scheme call.
  WRITE(frozen_qv_path,'(A,I0,A)') 'inputs',case_id,'.txt'
  OPEN(99,FILE=TRIM(frozen_qv_path),STATUS='OLD',ACTION='READ')
  DO k=kts,kte
    READ(99,*) qv(1,k,1)
  ENDDO
  CLOSE(99)
"""
        assert text.count('  hgt3d = 0.0')==1
        text=text.replace('  hgt3d = 0.0',insertion+'  hgt3d = 0.0')
        dst=out/(group+'_driver.f90');dst.write_text(text)
        subprocess.run([compiler,*common,'-ffp-contract=off',f'-I{out}',f'-I{base}','-c',str(dst),'-o',str(out/(group+'_driver.o'))],check=True,cwd=out)
        subprocess.run([compiler,*common,*deps,str(out/'thompson_fma.o'),str(out/(group+'_driver.o')),'-o',str(out/('oracle_'+group))],check=True,cwd=out)
    all_rows=[]
    for case in range(1,10):
        reference=(base if case<=6 else fresh)/f'case{case}.txt'
        ref=columns(reference)
        (out/f'inputs{case}.txt').write_text(''.join(f'{value:.17e}\n' for value in ref['QV_IN']))
        with (out/f'case{case}.txt').open('w') as stream:
            subprocess.run([str(out/('oracle_base' if case<=6 else 'oracle_fresh')),str(case)],cwd=out,stdout=stream,check=True)
        actual=columns(out/f'case{case}.txt')
        checks={name:struct.pack('<'+str(len(values))+'f',*values)==struct.pack('<'+str(len(values))+'f',*actual[name]) for name,values in ref.items() if name.endswith('_IN') or name in ('P','PII','DELZ')}
        assert all(checks.values()),(case,checks)
        all_rows.append(dict(case=case,identical_input_fields=checks))
    receipt=dict(scope='Unmodified whole original module host FMA vs original no-FMA; identical input/table work',source_sha256=expected,flags=common+['-ffp-contract=fast','-march=native'],caller_flags=common+['-ffp-contract=off'],wall_s=time.monotonic()-started,inputs=all_rows)
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path);p.add_argument('fresh',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.base,a.fresh,a.out)
