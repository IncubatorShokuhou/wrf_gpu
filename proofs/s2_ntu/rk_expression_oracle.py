"""Original WRF scalar-update expression on captured, assembled operands.

This isolates the assignment after tendency assembly and base-mass addition.
It does not claim a whole WRF caller or coupled forecast replay. REAL8 is a
promoted diagnostic for the FAST0 carry; original WRF's operator is REAL4.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

SOURCE=Path('<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F')


def build(folder,kind,contraction):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    text=SOURCE.read_text()
    routine=re.search(r'(?ims)^SUBROUTINE rk_update_scalar\s*\(.*?^END SUBROUTINE rk_update_scalar\s*$',text).group(0)
    expression=re.search(r'(?im)^\s*scalar_2\(i,k,j,im\)\s*=\s*[^\n]*&\n[^\n]*$',routine).group(0).strip()
    assert 'scalar_1(i,k,j,im)' in expression and 'dt*tendency(i,k,j)' in expression
    real='c_float' if kind=='r4' else 'c_double'
    code=f'''subroutine original(nx,ny,nz,dt,q,t,c1,c2,mo,mn,out) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: nx,ny,nz
real({real}),value :: dt
real({real}),intent(in) :: q(nx,nz,ny),t(nx,nz,ny),c1(nz),c2(nz),mo(nx,ny),mn(nx,ny)
real({real}),intent(out) :: out(nx,nz,ny)
real({real}) :: scalar_1(nx,nz,ny,1),scalar_2(nx,nz,ny,1),tendency(nx,nz,ny),muold(nx),munew(nx)
integer :: i,j,k,im
im=1
scalar_1(:,:,:,1)=q
tendency=t
do j=1,ny
muold=mo(:,j)
munew=mn(:,j)
do k=1,nz
do i=1,nx
{expression}
end do
end do
end do
out=scalar_2(:,:,:,1)
end subroutine
'''
    source=folder/'rk_expression.f90';source.write_text(code)
    library=folder/'rk_expression.so'
    command=['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran',
             '-O0' if contraction=='off' else '-O3','-march=native',
             '-ffp-contract='+contraction,'-fPIC','-shared','-ffree-line-length-none',str(source),'-o',str(library)]
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=60)
    manifest=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),expression=expression,
        expression_sha256=hashlib.sha256(expression.encode()).hexdigest(),compile=command,kind=kind,
        scope=__doc__)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return library


def evaluate(library,arrays,name,kind):
    dtype=np.float32 if kind=='r4' else np.float64
    real=ctypes.c_float if kind=='r4' else ctypes.c_double
    q=arrays[name+'_origin'];nz,ny,nx=q.shape
    volume=lambda a:np.asfortranarray(np.asarray(a,dtype).transpose(2,0,1))
    plane=lambda a:np.asfortranarray(np.asarray(a,dtype).T)
    inputs=[volume(q),volume(arrays[name+'_tendency']),
            np.asarray(arrays['c1'],dtype),np.asarray(arrays['c2'],dtype),
            plane(arrays['mu_old']),plane(arrays['mu_new'])]
    out=np.empty((nx,nz,ny),dtype,order='F')
    fn=ctypes.CDLL(str(library)).original
    fn.restype=None
    fn.argtypes=[ctypes.c_int]*3+[real]+[ctypes.c_void_p]*7
    fn(nx,ny,nz,float(arrays['dt']),*[a.ctypes.data for a in inputs],out.ctypes.data)
    return out.transpose(1,2,0).copy()
