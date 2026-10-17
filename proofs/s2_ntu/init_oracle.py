"""Observe original cold initialization before any artificial cloud/aerosol edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from build_oracle import FC,NK,CASES,driver
from port_scalar import SOURCE


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--kind',choices=('r4','r8'),default='r4');a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=driver();needle='  d=(/20.e-6'
    assert source.count(needle)==1
    before=source.index(needle)
    record="""  write(filename,'(a,i2.2,a)')'init_',c,'.bin'
  open(newunit=u,file=trim(filename),access='stream',form='unformatted',status='replace')
  write(u)p(1,1:nk,1),alt(1,1:nk,1),qv(1,1:nk,1),ph(1,:,1),phb(1,:,1),xland
  write(u)qdcn(1,1:nk,1),qtcn(1,1:nk,1),qccn(1,1:nk,1),qrcn(1,1:nk,1),qnin(1,1:nk,1)
  close(u)
  cycle
"""
    source=source[:before]+record+source[before:]
    (out/'driver.F90').write_text(source)
    subprocess.run([FC,'-E','-P','-cpp','-ffree-form',str(SOURCE),'-o',str(out/'original.F90')],check=True)
    (out/'error.F90').write_text('module module_wrf_error\ncontains\nsubroutine wrf_error_fatal(s)\ncharacter(*)::s\nprint *,s\nerror stop 1\nend subroutine\nend module\n')
    cmd=[FC,'-ffree-form','-ffree-line-length-none','-O0','-ffp-contract=off','-fno-range-check','-finit-real=zero','-finit-integer=0','-finit-logical=false','error.F90','original.F90','driver.F90','-o','oracle']
    if a.kind=='r8':cmd[1:1]=['-fdefault-real-8','-fdefault-double-8']
    subprocess.run(cmd,cwd=out,check=True)
    dtype=np.dtype('float32' if a.kind=='r4' else 'float64');fields={}
    with (out/'stdout.log').open('w') as log:
        for ccnty in range(1,5):
            subprocess.run([str(out/'oracle'),'1',str(len(CASES)),str(ccnty),'0'],cwd=out,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            for case in range(1,len(CASES)+1):
                values=np.fromfile(out/f'init_{case:02}.bin',dtype=dtype);offset=0
                for name,n in [('p',NK),('alt',NK),('qv',NK),('ph',NK+1),('phb',NK+1),('xland',1)]+[(n,NK) for n in ('qdcn','qtcn','qccn','qrcn','qnin')]:
                    fields[f'cc{ccnty}_c{case:02}_{name}']=values[offset:offset+n].copy();offset+=n
                assert offset==values.size
    assert all(np.isfinite(v).all() for v in fields.values())
    np.savez_compressed(out/'initialization.npz',**fields)
    result={'kind':a.kind,'compile':cmd,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'npz_sha256':hashlib.sha256((out/'initialization.npz').read_bytes()).hexdigest(),'ccnty':[1,2,3,4],'cases':len(CASES),'levels':NK,'nonfinite':0}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    for name in ('original.F90','error.F90','driver.F90','oracle','module_mp_ntu.mod','module_wrf_error.mod'):(out/name).unlink(missing_ok=True)
    for case in range(1,len(CASES)+1):(out/f'init_{case:02}.bin').unlink()


if __name__=='__main__':main()
