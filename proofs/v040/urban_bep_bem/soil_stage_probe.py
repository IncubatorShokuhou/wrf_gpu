"""Observe pristine BEP soil_temp stages; require whole-output byte identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from build_oracle import parse_dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);parser.add_argument('--gold',type=Path,required=True);args=parser.parse_args()
    assert os.environ['JAX_PLATFORMS']=='cpu';assert not Path('/tmp/wrf_gpu2_quiet').exists()
    source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_bep.F').read_text()
    a=source.index('      subroutine soil_temp(');b=source.index('      end subroutine soil_temp',a)
    body=source[a:b]
    body=body.replace('      integer iz\n','      integer iz\n      integer,save :: obs_count=0\n')
    at='      tsig=temp(nz)'
    capture='''      obs_count=obs_count+1
      write(*,'(A,2I6,8(1X,ES24.16E3))') 'SOILMETA ',obs_count,nz,rs,rl,press,dt,em,alb,sf,rcp_u
      do iz=1,nz
        write(*,'(A,2I6,4(1X,ES24.16E3))') 'SOILIN ',obs_count,iz,dz(iz),temp(iz),ala(iz),cs(iz)
      enddo
'''
    assert body.count(at)==1;body=body.replace(at,capture+at)
    body=body.replace('      return', '''      do iz=1,nz
        write(*,'(A,2I6,1X,ES24.16E3)') 'SOILOUT ',obs_count,iz,temp(iz)
      enddo
      write(*,'(A,I6,3(1X,ES24.16E3))') 'SOILDIAG ',obs_count,pt,rt,gf
      return''')
    altered=source[:a]+body+source[b:]
    (args.build/'module_sf_bep_observed.F').write_text(altered)
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1','-fallow-argument-mismatch','-finit-integer=0','-fcheck=bounds']
    subprocess.run([fc,*flags,'-c','module_sf_bep_observed.F'],cwd=args.build,check=True,timeout=60)
    objects=[p.name for p in args.build.glob('*.o') if p.name not in ('module_sf_bep.o','module_sf_bep_observed.o')]
    subprocess.run([fc,*flags,'-o','observed',str(Path(__file__).resolve().parent/'bep_bem_oracle.F90'),'module_sf_bep_observed.o',*objects],cwd=args.build,check=True,timeout=60)
    run=args.build/'run2_ac_on'
    with (run/'observed.txt').open('w') as out:
        subprocess.run([str(args.build.resolve()/'observed'),'fp32',str(Path(__file__).resolve().parent/'regimes.txt')],cwd=run,stdout=out,check=True,timeout=60)
    parse_dump(run/'observed.txt',run/'observed.npz')
    with np.load(run/'observed.npz') as observed,np.load(args.gold) as gold:
        assert observed.files==gold.files
        for field in gold.files:assert np.array_equal(observed[field],gold[field]),field
    observations={}
    for line in (run/'observed.txt').read_text().splitlines():
        if line.startswith('SOIL'):
            fields=line.split();kind=fields[0];index=int(fields[1]);values=[float(v) for v in fields[2:]]
            observations.setdefault(index,{}).setdefault(kind,[]).append(values)
    result={'observed_original_output_arrays_BYTE':len(gold.files),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'observed_source_sha256':hashlib.sha256(altered.encode()).hexdigest(),'calls':observations}
    (args.build/'soil_stages.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BYTE original arrays; observed soil calls',len(observations))


if __name__=='__main__':main()
