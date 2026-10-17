"""Independent original NCAR wrf-python CALCDBZ reference (no Python port calls).

Stage fortran/{wrf_constants.f90,wrf_user_dbz.f90} from wrf-python tag v1.3.4.1
in --source; record exact hashes. Compile only in --build, never in source.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import numpy as np


def generate(source, build, output):
    build.mkdir(parents=True,exist_ok=True)
    paths=[source/name for name in ['wrf_constants.f90','wrf_user_dbz.f90']]
    driver='''
program oracle
 implicit none
 integer i,iv,skin,snowpresent,n
 double precision a(1,1,1,6),z(1,1,1)
 read(*,*)n
 do i=1,n
  read(*,*)a,iv,skin,snowpresent
  call CALCDBZ(a(:,:,:,1),a(:,:,:,2),a(:,:,:,3),a(:,:,:,4),a(:,:,:,5),a(:,:,:,6), &
               snowpresent,iv,skin,z,1,1,1)
  write(*,'(ES24.16)')z
 enddo
end program
'''
    (build/'driver.f90').write_text(driver)
    cmd=[os.environ.get('FC','gfortran'),'-O2','-fno-fast-math','-ffp-contract=off',*[str(p) for p in paths],'driver.f90','-o','oracle']
    subprocess.run(cmd,cwd=build,check=True)
    rows=[]
    for t in [250.,273.15,290.]:
        for amount in [0.,1e-16,1e-8,1e-4,.003]:
            for iv,skin,snow in [(0,0,1),(1,0,1),(0,1,1),(1,1,0)]:
                rows.append([85000.,t,.008,amount,amount*.5,amount*.2,iv,skin,snow])
    inp=str(len(rows))+'\n'+'\n'.join(' '.join(map(str,r)) for r in rows)+'\n'
    ref=subprocess.run([str(build/'oracle')],input=inp,capture_output=True,text=True,check=True,timeout=30)
    z=np.fromstring(ref.stdout,sep=' ')
    output.mkdir(parents=True,exist_ok=True)
    (output/'rip_oracle.json').write_text(json.dumps({'upstream':'NCAR/wrf-python v1.3.4.1','sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'compiler_command':cmd,'rows':rows,'dbz':z.tolist()},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','build','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();generate(a.source,a.build,a.output)
