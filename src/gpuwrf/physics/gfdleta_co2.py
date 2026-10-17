"""Once-at-init, source-preserving WRF GFDL CO2 provider (no JAX imports).

The bundled native algorithm is not called from a forecast step. GNU Fortran
is an optional dependency of ra=99 only, required to build its cached helper.
"""
from __future__ import annotations

from functools import lru_cache
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

_ASSETS=Path(__file__).with_name('gfdleta_co2_assets')
_CO2=('co251','cdt51','cdt58','c2d51','c2d58','co258','stemp','gtemp',
      'co231','co238','c2d31','c2d38','cdt31','cdt38','co271','co278',
      'c2d71','c2d78','cdt71','cdt78','co2m51','co2m58','cdtm51','cdtm58','c2dm51','c2dm58')
_MATRICES={'co251','cdt51','cdt58','c2d51','c2d58','co258'}
_LAYERS={'co2m51','co2m58','cdtm51','cdtm58','c2dm51','c2dm58'}


def _executable(cache_directory=None):
    compiler=shutil.which(os.environ.get('FC','gfortran'))
    if compiler is None:
        raise RuntimeError('GFDL ra=99 initialization needs GNU Fortran (install gfortran or set FC)')
    version=subprocess.run([compiler,'--version'],check=True,timeout=10,capture_output=True).stdout
    identity=hashlib.sha256(version+str(Path(compiler).resolve()).encode())
    files=sorted(p for p in _ASSETS.iterdir() if p.is_file())
    for path in files:identity.update(path.name.encode()+path.read_bytes())
    base=Path(cache_directory or os.environ.get('GPUWRF_GFDL_INIT_CACHE',
        Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'gpuwrf/gfdl_init'))
    directory=base/identity.hexdigest();directory.mkdir(parents=True,exist_ok=True)
    executable=directory/'co2_init'
    with (directory/'build.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not executable.is_file():
            with tempfile.TemporaryDirectory(prefix='build-',dir=directory) as temporary:
                build=Path(temporary)
                for name in ('module_model_constants.F','co2_host.f90','serial_support.f90'):
                    shutil.copyfile(_ASSETS/name,build/name)
                command=[compiler,'-O2','-cpp','-ffree-form','-ffree-line-length-none',
                    '-fallow-argument-mismatch','-DRWORDSIZE=4','-DIWORDSIZE=4',
                    '-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffp-contract=off',
                    'module_model_constants.F','serial_support.f90','co2_host.f90','-o','co2_init']
                result=subprocess.run(command,cwd=build,timeout=120,capture_output=True,
                                      env=dict(os.environ,TMPDIR=temporary))
                if result.returncode:
                    raise RuntimeError('GFDL CO2 initializer compile failed: '+result.stderr.decode(errors='replace'))
                os.replace(build/'co2_init',executable)
    return executable


@lru_cache(maxsize=8)
def _tables(full_bytes,half_bytes,ptop_bytes,cache_directory):
    full=np.frombuffer(full_bytes,np.float32);half=np.frombuffer(half_bytes,np.float32)
    nz=len(half);executable=_executable(cache_directory)
    with tempfile.TemporaryDirectory(prefix='grid-',dir=executable.parent) as temporary:
        directory=Path(temporary)
        for name in ('tr49t85','tr49t67','tr67t85'):(directory/name).symlink_to((_ASSETS/name).resolve())
        source=directory/'grid.bin';target=directory/'tables.bin'
        with source.open('wb') as stream:
            np.asarray(nz,np.int32).tofile(stream);stream.write(ptop_bytes)
            stream.write(full_bytes);stream.write(half_bytes)
        result=subprocess.run([str(executable.resolve()),str(source),str(target)],cwd=directory,
                              timeout=30,capture_output=True)
        if result.returncode:
            raise RuntimeError('GFDL CO2 initialization failed: '+result.stderr.decode(errors='replace'))
        with target.open('rb') as stream:
            if np.fromfile(stream,np.int32,1).tolist()!=[nz]:raise ValueError('Invalid GFDL provider dimension')
            domain={'ltop':np.fromfile(stream,np.int32,3),'ptopc':np.fromfile(stream,np.float32,4)}
            co2={}
            for name in _CO2:
                shape=(nz+1,nz+1) if name in _MATRICES else (nz,) if name in _LAYERS else (nz+1,)
                co2[name]=np.fromfile(stream,np.float32,int(np.prod(shape))).reshape(shape,order='F')
            if stream.read(1):raise ValueError('Trailing GFDL provider bytes')
    for value in (*domain.values(),*co2.values()):
        if not np.isfinite(value).all():raise ValueError('Nonfinite GFDL CO2 provider')
        value.setflags(write=False)
    return co2,domain


def co2_tables(eta_interfaces,eta_midpoints,top_pressure_pa,*,cache_directory=None):
    """Return read-only host CO2/domain dictionaries for one actual WRF grid."""
    full=np.asarray(eta_interfaces,np.float32);half=np.asarray(eta_midpoints,np.float32)
    ptop=np.asarray(top_pressure_pa,np.float32)
    if (full.ndim!=1 or half.ndim!=1 or len(full)!=len(half)+1 or len(half)<3
        or ptop.shape or not np.isfinite(ptop) or ptop<=0 or ptop>=100000
        or not np.isfinite(full).all() or not np.isfinite(half).all()
        or full[0]!=1 or full[-1]!=0 or np.any(np.diff(full)>=0)
        or np.any(half>=full[:-1]) or np.any(half<=full[1:])):
        raise ValueError('GFDL CO2 initialization requires finite descending WRF sigma interfaces/midpoints and p_top')
    co2,domain=_tables(full.tobytes(),half.tobytes(),ptop.tobytes(),
                       str(Path(cache_directory).resolve()) if cache_directory is not None else None)
    # Do not expose mutable cached dictionaries to callers.
    return dict(co2),dict(domain)
