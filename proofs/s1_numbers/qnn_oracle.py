"""Stage and compile unchanged pristine WRF flow_dep_bdy[_qnn], REAL4."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

SOURCE = Path('<USER_HOME>/src/wrf_pristine/WRF/share/module_bc.F')


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text()
    names = ('flow_dep_bdy', 'flow_dep_bdy_qnn')
    bodies = {n: re.search(r'(?ims)^\s*SUBROUTINE '+n+r'\b.*?^\s*END SUBROUTINE '+n+r'\b', text).group()
              for n in names}
    head = '''module numbers_bc_oracle
use iso_c_binding
implicit none
type grid_config_rec_type
logical :: periodic_x=.false.
end type
contains
'''
    wrapper = '''
subroutine boundary(nx,ny,nz,spec,qnn,ccn,q,u,v) bind(c)
integer(c_int),value :: nx,ny,nz,spec,qnn
real(c_float),value :: ccn
real(c_float) :: q(nx+1,nz,ny+1),u(nx+1,nz,ny+1),v(nx+1,nz,ny+1)
type(grid_config_rec_type) :: cfg
if(qnn/=0)then
call flow_dep_bdy_qnn(q,u,v,cfg,spec,ccn, &
 1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz, &
 1,nx+1,1,ny+1,1,nz,1,nx,1,ny,1,nz)
else
call flow_dep_bdy(q,u,v,cfg,spec, &
 1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz, &
 1,nx+1,1,ny+1,1,nz,1,nx,1,ny,1,nz)
endif
end subroutine
end module
'''
    staged = directory/'boundary.f90'
    staged.write_text(head+'\n'.join(bodies.values())+wrapper)
    library = directory/'boundary.so'
    cmd = [os.environ.get('FC', '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'),
           '-O2', '-fPIC', '-shared', '-ffree-line-length-none', '-ffp-contract=off', '-fcheck=bounds',
           '-J', str(directory), str(staged), '-o', str(library)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    (directory/'manifest.json').write_text(json.dumps({
        'source': str(SOURCE), 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'routine_sha256': {n: hashlib.sha256(b.encode()).hexdigest() for n, b in bodies.items()},
        'command': cmd, 'REAL_bytes': 4, 'scope': 'Specified nonperiodic full tile; exact i,k,j staggering; '
        'caller ru_m/rv_m signs, spec_zone and ccn_conc from solve_em.F:2983. No wrf.exe executed.',
    }, indent=2)+'\n')
    return library


def evaluate(library, q, u, v, *, spec=1, qnn=False, ccn=1e8):
    nz, ny, nx = q.shape
    def volume(a):
        a = np.asarray(a, np.float32)
        return np.asfortranarray(np.pad(a, ((0, 0), (0, ny+1-a.shape[1]), (0, nx+1-a.shape[2])),
                                       mode='edge').transpose(2, 0, 1))
    f, ru, rv = (volume(a) for a in (q, u, v))
    fn = ctypes.CDLL(str(library)).boundary
    fn.argtypes = [ctypes.c_int]*5+[ctypes.c_float]+[ctypes.c_void_p]*3
    fn(nx, ny, nz, spec, int(qnn), ccn, f.ctypes.data, ru.ctypes.data, rv.ctypes.data)
    return f[:nx, :, :ny].transpose(1, 2, 0).copy()
