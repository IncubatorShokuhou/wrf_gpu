"""Original WRF option-1 length routine with only ALP1 supplied externally.

The reference wrapper uses the original plume and stability operands from the
existing real-column fixture. No pristine source or object is modified.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF/phys')
WRAPPER = """
subroutine lengths(n,nb,factor,dx,xland,dz,zw,rmol,fltv,u,v,qke,dtv,pblh,theta,psig,cloud,edmf_w,edmf_a,out) bind(C)
use iso_c_binding
use @MODULE@, only: mym_length @IMPORT@
implicit none
integer(c_int),value::n,nb
real(c_float),value::factor,dx
real(c_float),intent(in)::xland(nb),dz(n,nb),zw(n+1,nb),rmol(nb),fltv(nb),u(n,nb),v(n,nb)
real(c_float),intent(in)::qke(n,nb),dtv(n,nb),pblh(nb),theta(n,nb),psig(nb),cloud(n,nb),edmf_w(n,nb),edmf_a(n,nb)
real(c_float),intent(out)::out(n,2,nb)
real(c_float)::zeros(n)
integer::col
zeros=0.
@SET@
do col=1,nb
 call mym_length(1,n,xland(col),dz(:,col),dx,zw(:,col),rmol(col),0._c_float,fltv(col),0._c_float, &
  zeros,zeros,u(:,col),v(:,col),qke(:,col),dtv(:,col),out(:,2,col),pblh(col),theta(:,col), &
  out(:,1,col),psig(col),cloud(:,col),1,edmf_w(:,col),edmf_a(:,col),zeros,zeros)
enddo
end subroutine
"""


def build(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = (WRF / 'module_bl_mynnedmf.F').read_text()
    header = re.search(r'(?ims)^ *module module_bl_mynnedmf\b.*?^ *contains\b', source).group(0)
    header = re.sub(r'(?i)module module_bl_mynnedmf\b', 'module w44_length', header, count=1)
    header = re.sub(r'(?i)contains\s*$', 'real(kind_phys)::w44_alp1\ncontains', header)
    bodies = {name: re.search(r'(?ims)^ *subroutine +' + name + r'\b.*?^ *end subroutine +' + name + r'\b', source).group(0)
              for name in ('mym_length', 'boulac_length')}
    assert bodies['mym_length'].count('alp1  = 0.23') == 1
    adapted = bodies['mym_length'].replace('alp1  = 0.23', 'alp1  = w44_alp1')
    extracted = header + '\n' + adapted + '\n' + bodies['boulac_length'] + '\nend module\n'
    (directory / 'length.F90').write_text(extracted)
    compiler = os.environ.get('FC', '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    paths, commands = {}, []
    for label in ('original', 'runtime'):
        wrapper = WRAPPER.replace('@MODULE@', 'module_bl_mynnedmf' if label == 'original' else 'w44_length')
        wrapper = wrapper.replace('@IMPORT@', '' if label == 'original' else ',w44_alp1')
        wrapper = wrapper.replace('@SET@', '' if label == 'original' else 'w44_alp1=factor')
        path = directory / (label + '.f90')
        path.write_text(wrapper)
        core = WRF / 'module_bl_mynnedmf.o' if label == 'original' else directory / 'length.F90'
        output = directory / (label + '.so')
        command = [compiler, '-cpp', '-O2', '-fPIC', '-shared', '-ffree-line-length-none', '-ffp-contract=off',
                   '-I' + str(WRF), '-I' + str(directory), '-J' + str(directory), str(core), str(path),
                   str(WRF / 'module_bl_mynnedmf_common.o'), '-o', str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, cwd=directory)
        paths[label] = output
        commands.append(command)
    (directory / 'manifest.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'original_bodies_sha256': {name: hashlib.sha256(body.encode()).hexdigest() for name, body in bodies.items()},
        'only_arithmetic_source_change': 'CASE(1) alp1 = 0.23 -> alp1 = w44_alp1',
        'original_objects': {name: hashlib.sha256((WRF / name).read_bytes()).hexdigest()
                             for name in ('module_bl_mynnedmf.o', 'module_bl_mynnedmf_common.o')},
        'commands': commands, 'kind_phys': 4, 'mixlength': 1, 'downdraft': False,
    }, indent=2))
    return paths


def evaluate(library, row, alp1=.23):
    v, f = np.asarray(row['values'], np.float32), np.asarray(row['flux'], np.float32)
    nb, nz = v.shape[:2]
    args = [f[:, 4], v[:, :, 8], row['zw'], row['rmol'], f[:, 3], v[:, :, 0], v[:, :, 1],
            np.float32(2) * v[:, :, 5], row['dtv'], row['pblh'], v[:, :, 3], row['psig_bl'],
            row['cldfra_bl'], row['edmf_w'], row['edmf_a']]
    args = [np.ascontiguousarray(a, np.float32) for a in args]
    assert all(np.isfinite(a).all() for a in args)
    out = np.empty((nb, 2, nz), np.float32)
    lib = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    lib.lengths.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float] + [pointer] * 16
    lib.lengths.restype = None
    lib.lengths(nz, nb, alp1, 1000., *args, out)
    assert np.isfinite(out).all()
    return out
