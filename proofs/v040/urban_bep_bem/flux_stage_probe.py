"""Calibrate WRF-only flux operands without changing original numerical output.

This records inputs/coefficients for a proposed roundoff budget. It changes
neither candidate nor registered gates and is not a passing qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from build_oracle import parse_dump


def observe(source,routine,tag,fields):
    a=source.index('      subroutine '+routine+'(')
    b=source.index('      end subroutine '+routine,a)
    body=source[a:b]
    assert body.count('      implicit none')==1
    body=body.replace('      implicit none','      implicit none\n      integer,save :: flux_observation_count=0')
    at='      return'
    assert body.count(at)==1
    capture="      flux_observation_count=flux_observation_count+1\n"
    capture+=f"      write(*,'(A,I8,*(1X,ES24.16E3))') '{tag} ',flux_observation_count,{','.join(fields)}\n"
    body=body.replace(at,capture+at)
    return source[:a]+body+source[b:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--gold',type=Path,required=True)
    a=p.parse_args();assert os.environ['JAX_PLATFORMS']=='cpu'
    pristine=Path('<USER_HOME>/src/wrf_pristine/WRF/phys');here=Path(__file__).resolve().parent
    sources={name:(pristine/name).read_text() for name in ('module_sf_bep.F','module_sf_bep_bem.F')}
    specs={
        'module_sf_bep.F':[('flux_wall','BEPWALL',['ua','va','pt','da','ptw','drst','dt','hc','tvb','uva','vva','uvb','vvb'])],
        'module_sf_bep_bem.F':[
            ('flux_wall','BEMWALL',['ua','va','pt','da','ptw','ptwin','drst','dt','cdrag','hc','hcwin','sfw','sfwin']),
            ('flux_flat','BEMFLAT',['dz','z0','ua','va','pt','pt0','ptg','da','qv','pr','aa','utot','fh','rr','qvsg','resg','sf','lf'])]}
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1','-fallow-argument-mismatch','-finit-integer=0','-fcheck=bounds']
    def run(command,**kwargs):
        if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit(3)
        return subprocess.run(command,cwd=a.build,check=True,timeout=60,**kwargs)
    modified={}
    for name,text in sources.items():
        altered=text
        for routine,tag,fields in specs[name]:altered=observe(altered,routine,tag,fields)
        observed=name.replace('.F','_flux_observed.F');(a.build/observed).write_text(altered)
        run([fc,*flags,'-c',observed]);modified[name]=hashlib.sha256(altered.encode()).hexdigest()
    excluded={'module_sf_bep.o','module_sf_bep_bem.o'}
    objects=[x.name for x in a.build.glob('*.o') if x.name not in excluded and 'observed' not in x.name]
    for option in (2,3):
        run([fc,*flags,*(['-DWITH_BEM'] if option==3 else []),'-o',f'flux_observed{option}',str(here/'bep_bem_oracle.F90'),
            'module_sf_bep_flux_observed.o','module_sf_bep_bem_flux_observed.o',*objects])
    receipt={'original_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},'observed_sha256':modified,'arms':{}}
    for option,ac in [(2,'ac_on'),(3,'ac_on'),(3,'ac_off')]:
        directory=a.build/f'run{option}_{ac}'
        path=directory/'flux_observed.txt'
        with path.open('w') as f:
            if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit(3)
            subprocess.run([str(a.build.resolve()/f'flux_observed{option}'),'fp32',str(here/'regimes.txt')],
                cwd=directory,stdout=f,check=True,timeout=60)
        parse_dump(path,directory/'flux_observed.npz')
        with np.load(directory/'flux_observed.npz') as observed,np.load(a.gold/f'urban{option}_fp32_{ac}.npz') as gold:
            assert observed.files==gold.files
            for field in gold.files:
                assert observed[field].shape==gold[field].shape,(field,observed[field].shape,gold[field].shape)
                assert np.array_equal(observed[field],gold[field]),field
            count=len(gold.files)
        captures={}
        for line in path.read_text().splitlines():
            for specs_for_source in specs.values():
                for _,tag,fields in specs_for_source:
                    if line.startswith(tag+' '):
                        values=line.split();captures.setdefault(tag,[]).append(dict(zip(['call']+fields,map(float,values[1:]))))
        receipt['arms'][f'urban{option}_{ac}']={'original_arrays_BYTE':count,'operands':captures}
    target=a.build/'flux_stages.json';target.write_text(json.dumps(receipt,indent=2)+'\n')
    print({k:{'BYTE':v['original_arrays_BYTE'],'calls':{tag:len(rows) for tag,rows in v['operands'].items()}} for k,v in receipt['arms'].items()})


if __name__=='__main__':main()
