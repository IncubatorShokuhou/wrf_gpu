"""Compile unchanged original add_a2a/add_a2c bodies, four ownership regimes."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import numpy as np

work=Path(sys.argv[1]);work.mkdir(parents=True,exist_ok=True)
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_physics_addtendc.F').read_bytes()
names=('add_a2a','add_a2c_u','add_a2c_v')
bodies=[re.search(r'^SUBROUTINE '+n+r'\(.*?^END SUBROUTINE '+n+r'\s*$',
                  source.decode(),re.M|re.S)[0] for n in names]
module='''module original_addtend
type grid_config_rec_type
logical :: specified=.true.,nested=.false.,periodic_x=.false.
end type
contains
'''+ '\n'.join(bodies)+'\nend module\n'
driver='''program driver
use original_addtend
implicit none
integer,parameter :: nx=6,ny=5,nz=3
integer :: i,j,k,arm
type(grid_config_rec_type) :: config
real :: a(0:nx+1,1:nz,0:ny+1),u(0:nx+1,1:nz,0:ny+1),v(0:nx+1,1:nz,0:ny+1)
real :: r(0:nx+1,1:nz,0:ny+1)
do arm=1,4
config%specified=mod(arm,2)==1
config%nested=.not.config%specified
config%periodic_x=arm>2
a=0.;u=0.;v=0.;r=1.
call add_a2a(a,r,config,1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz,1,nx+1,1,ny+1,1,nz)
call add_a2c_u(u,r,config,1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz,1,nx+1,1,ny+1,1,nz)
call add_a2c_v(v,r,config,1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz,1,nx+1,1,ny+1,1,nz)
write(*,'(*(F3.0))') (((a(i,k,j),i=1,nx),k=1,nz),j=1,ny)
write(*,'(*(F3.0))') (((u(i,k,j),i=1,nx+1),k=1,nz),j=1,ny)
write(*,'(*(F3.0))') (((v(i,k,j),i=1,nx),k=1,nz),j=1,ny+1)
end do
end program
'''
(work/'original.f90').write_text(module+driver)
compiler='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
subprocess.run([compiler,'-O2','-ffree-form','-ffree-line-length-none','-o','oracle','original.f90'],cwd=work,check=True,timeout=60)
raw=subprocess.check_output([str(work.resolve()/'oracle')],cwd=work,timeout=30)
(work/'stdout.txt').write_bytes(raw);lines=raw.decode().splitlines();assert len(lines)==12
arrays={}
for i,arm in enumerate(('specified','nested','specified_periodic','nested_periodic')):
    for j,(field,ny,nx) in enumerate((('scalar',5,6),('u',5,7),('v',6,6))):
        arrays[f'{arm}_{field}']=np.fromstring(lines[3*i+j],sep=' ').reshape(ny,3,nx).transpose(1,0,2).astype(np.float32)
np.savez_compressed(work/'boundary.npz',**arrays)
manifest={'original_source_sha256':hashlib.sha256(source).hexdigest(),
 'body_sha256':{n:hashlib.sha256(b.encode()).hexdigest() for n,b in zip(names,bodies)},
 'fixture_sha256':hashlib.sha256((work/'boundary.npz').read_bytes()).hexdigest(),
 'original_bodies_unchanged':True,'shape_nz_ny_nx':[3,5,6],
 'counts':{k:int(np.count_nonzero(v)) for k,v in arrays.items()}}
(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
