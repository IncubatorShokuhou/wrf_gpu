"""Unchanged WRF interp_fcn_sint + SINT, with a REAL4 full-field C adapter."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {name: (WRF / name).read_text() for name in ('share/interp_fcn.F', 'share/sint.F')}
    routines = {}
    for name, source in (('interp_fcn_sint', sources['share/interp_fcn.F']), ('SINT', sources['share/sint.F'])):
        match = re.search(r'(?ims)^\s*SUBROUTINE '+name+r'\b.*?^\s*END(?:\s+SUBROUTINE(?:\s+'+name+r')?)?\s*$', source)
        if match is None:
            raise ValueError(name)
        routines[name] = match.group(0)
    wrapper = '''module module_configure
end module
module nest_start_adapter
use iso_c_binding
implicit none
contains
subroutine nest_start(px,py,nx,ny,nz,ipos,jpos,ratio,parent,child) bind(c)
integer(c_int),value :: px,py,nx,ny,nz,ipos,jpos,ratio
real(c_float) :: parent(-2:px+3,nz,-2:py+3),child(nx+1,nz,ny+1)
integer :: imask(nx+1,ny+1)
imask=1
child=-999.
call interp_fcn_sint(parent, &
 1,px+1,1,nz+1,1,py+1,-2,px+3,1,nz,-2,py+3,1,px+1,1,nz,1,py+1, &
 child,1,nx+1,1,nz+1,1,ny+1,1,nx+1,1,nz,1,ny+1,1,nx+1,1,nz,1,ny+1, &
 3,imask,.false.,.false.,ipos,jpos,ratio,ratio)
end subroutine
end module
'''
    path = directory / 'nest_start.f90'
    path.write_text(wrapper+'\n'+'\n'.join(routines.values()))
    command = ['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran', '-O2', '-shared', '-fPIC',
               '-ffree-line-length-none', '-ffp-contract=off', '-fcheck=bounds', '-J', str(directory),
               str(path), '-o', str(directory / 'nest_start.so')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr)
    (directory / 'manifest.json').write_text(json.dumps({
        'source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
        'routine_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in routines.items()},
        'REAL_bytes': 4, 'command': command,
        'caller': 'med_nest_initial -> med_interp_domain -> interp_domain_em_part3; uncoupled mass-grid d fields; input stream override follows',
    }, indent=2)+'\n')
    return directory / 'nest_start.so'


def interpolate(library, parent, *, child_shape, start=(6, 7), ratio=3):
    nz, py, px = parent.shape
    ny, nx = child_shape
    p = np.asfortranarray(np.pad(np.asarray(parent, np.float32), ((0, 0), (3, 3), (3, 3)), mode='edge').transpose(2, 0, 1))
    q = np.zeros((nx+1, nz, ny+1), np.float32, order='F')
    function = ctypes.CDLL(str(library)).nest_start
    function.argtypes = [ctypes.c_int]*8 + [ctypes.c_void_p]*2
    function.restype = None
    function(px, py, nx, ny, nz, *start, ratio, p.ctypes.data, q.ctypes.data)
    return q[:nx, :, :ny].transpose(1, 2, 0).copy()
