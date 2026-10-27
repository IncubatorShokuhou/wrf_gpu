"""Unchanged WRF smag2d_km with REAL c_s supplied through its config argument."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F")
    text = source.read_text()
    body = re.search(r"(?ims)^\s*SUBROUTINE smag2d_km\b.*?^END SUBROUTINE smag2d_km\b", text).group(0)
    constants = Path("<USER_HOME>/src/wrf_pristine/WRF/share/module_model_constants.F").read_text()
    assert re.search(r"(?i)\bprandtl\s*=\s*1\.\s*/\s*3\.", constants)
    header = """module w44_smag
use iso_c_binding
implicit none
real,parameter::prandtl=1./3.
type grid_config_rec_type
 logical::open_xs=.false.,open_xe=.false.,open_ys=.false.,open_ye=.false.
 logical::specified=.true.,nested=.false.,periodic_x=.false.
 integer::diff_opt=1
 real::c_s=.25
end type
contains
"""
    wrapper = """
subroutine smag(nx,ny,nz,cs,dx,dy,fields,maps,out) bind(C)
integer(c_int),value::nx,ny,nz
real(c_float),value::cs,dx,dy
real(c_float),intent(in)::fields(0:nx+1,1:nz+1,0:ny+1,6),maps(0:nx+1,0:ny+1,2)
real(c_float),intent(out)::out(0:nx+1,1:nz+1,0:ny+1,4)
type(grid_config_rec_type)::cfg
cfg%c_s=cs
out=0.
call smag2d_km(cfg,out(:,:,:,1),out(:,:,:,2),out(:,:,:,3),out(:,:,:,4), &
 fields(:,:,:,1),fields(:,:,:,2),fields(:,:,:,3),fields(:,:,:,4),dx,dy, &
 maps(:,:,1),maps(:,:,2),fields(:,:,:,5),fields(:,:,:,6), &
 1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz)
end subroutine
end module
"""
    path = directory / "smag.f90"
    path.write_text(header + body + wrapper)
    output = directory / "smag.so"
    command = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"),
               "-O2", "-fPIC", "-shared", "-ffree-line-length-none", "-ffp-contract=off", "-fcheck=bounds",
               "-J" + str(directory), str(path), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, cwd=directory)
    (directory / "manifest.json").write_text(json.dumps({"source": str(source),
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "routine_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "constants_sha256": hashlib.sha256(constants.encode()).hexdigest(),
        "caller": {"diff_opt": 1, "km_opt": 4, "specified": True, "real_bytes": 4},
        "command": command}, indent=2))
    return output


def evaluate(library, d11, d22, d12, mx, my, *, dx, dy, c_s):
    nz, ny, nx = d11.shape
    fields = np.zeros((nx + 2, nz + 1, ny + 2, 6), np.float32, order="F")
    for i, value in enumerate((d11, d22, d12)):
        fields[1:nx + 1, :nz, 1:ny + 1, i] = np.asarray(value).transpose(2, 0, 1)
    fields[..., 3] = 1.
    maps = np.ones((nx + 2, ny + 2, 2), np.float32, order="F")
    maps[1:nx + 1, 1:ny + 1, 0] = mx.T
    maps[1:nx + 1, 1:ny + 1, 1] = my.T
    out = np.empty((nx + 2, nz + 1, ny + 2, 4), np.float32, order="F")
    lib = ctypes.CDLL(str(library))
    lib.smag.argtypes = [ctypes.c_int] * 3 + [ctypes.c_float] * 3 + [ctypes.c_void_p] * 3
    lib.smag(nx, ny, nz, c_s, dx, dy, fields.ctypes.data, maps.ctypes.data, out.ctypes.data)
    assert np.isfinite(out).all()
    return tuple(out[1:nx + 1, :nz, 1:ny + 1, i].transpose(1, 2, 0).copy() for i in (0, 2))
