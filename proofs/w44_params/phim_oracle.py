"""Pristine WRF phim with only its two selected coefficients supplied externally.

Require default bytes to agree with the original WRF object before using the
extracted function for changed coefficients. The function's arithmetic remains
verbatim; only the local am_st PARAMETER declaration becomes module storage.
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
subroutine phi(n,z,unst,stable,out) bind(C)
use iso_c_binding
use @MODULE@, only: phim @IMPORT@
implicit none
integer(c_int),value::n
real(c_float),intent(in)::z(n)
real(c_float),value::unst,stable
real(c_float),intent(out)::out(n)
integer::i
@SET@
do i=1,n
 out(i)=phim(z(i))
enddo
end subroutine
"""


def build(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = (WRF / 'module_bl_mynnedmf.F').read_text()
    body = re.search(r'(?ims)^ *function phim\(zet\).*?^ *end function phim\b', source).group(0)
    declaration = 'real(kind_phys), parameter :: am_st=6.1, bm_st=2.5, rbm_st=1./bm_st'
    assert body.count(declaration) == 1
    assert 'cphm_unst=16.0' in source
    adapted = body.replace(declaration, 'real(kind_phys), parameter :: bm_st=2.5, rbm_st=1./bm_st')
    extracted = ('module w44_phim\nuse module_bl_mynnedmf_common, only: kind_phys,one,zero\n'
                 'implicit none\nreal(kind_phys)::cphm_unst,am_st\ncontains\n' + adapted +
                 '\nend module\n')
    (directory / 'phim.f90').write_text(extracted)
    compiler = os.environ.get('FC', '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    commands, paths = [], {}
    for label in ('original', 'runtime'):
        wrapper = WRAPPER.replace('@MODULE@', 'module_bl_mynnedmf' if label == 'original' else 'w44_phim')
        wrapper = wrapper.replace('@IMPORT@', '' if label == 'original' else ',cphm_unst,am_st')
        wrapper = wrapper.replace('@SET@', '' if label == 'original' else 'cphm_unst=unst;am_st=stable')
        path = directory / (label + '.f90')
        path.write_text(wrapper)
        core = WRF / 'module_bl_mynnedmf.o' if label == 'original' else directory / 'phim.f90'
        output = directory / (label + '.so')
        command = [compiler, '-O2', '-fPIC', '-shared', '-ffree-line-length-none', '-ffp-contract=off',
                   '-I' + str(WRF), '-I' + str(directory), '-J' + str(directory), str(core),
                   str(path), str(WRF / 'module_bl_mynnedmf_common.o'), '-o', str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, cwd=directory)
        commands.append(command)
        paths[label] = output
    (directory / 'manifest.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'original_body_sha256': hashlib.sha256(body.encode()).hexdigest(),
        'adapted_body_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'only_body_change': {'from': declaration, 'to': 'real(kind_phys), parameter :: bm_st=2.5, rbm_st=1./bm_st'},
        'original_objects': {name: hashlib.sha256((WRF / name).read_bytes()).hexdigest()
                             for name in ('module_bl_mynnedmf.o', 'module_bl_mynnedmf_common.o')},
        'commands': commands, 'kind_phys': 4,
    }, indent=2))
    return paths


def evaluate(library, zet, cphm_unst=16., am_st=6.1):
    values = np.ascontiguousarray(zet, np.float32)
    assert values.ndim == 1 and np.isfinite(values).all()
    lib = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    lib.phi.argtypes = [ctypes.c_int, pointer, ctypes.c_float, ctypes.c_float, pointer]
    lib.phi.restype = None
    out = np.empty_like(values)
    lib.phi(values.size, values, cphm_unst, am_st, out)
    assert np.isfinite(out).all()
    return out
