"""Private REAL4 WRF oracle: unchanged calc_cq and phy_prep pressure loops.

Array wiring mirrors module_em and first_rk_step_part1: the full Registry
moist array (dummy index 1, PARAM_FIRST_SCALAR=2), MUT, C1H/C2H and DNW.
No microphysics or full CPU-WRF trajectory claim is made by this oracle.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np


def build_oracle(tree, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(tree) / "data/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
    source = path.read_text()
    start = source.index("SUBROUTINE calc_cq (")
    end = source.index("END SUBROUTINE calc_cq", start) + len("END SUBROUTINE calc_cq")
    cq = source[start:end]
    start = source.index("! calculate hydrostatic pressure at both full and half levels")
    end = source.index("END SUBROUTINE phy_prep", start)
    hydro = source[start:end]
    wrapper = '''module original_moist
use iso_c_binding
implicit none
integer, parameter :: PARAM_FIRST_SCALAR=2
contains
''' + cq + '''
subroutine evaluate(nx,ny,nz,n_moist,moist,mut,c1,c2,dnw,p_top,p_hyd_w,p_hyd,cqw) bind(c)
integer(c_int), value :: nx,ny,nz,n_moist
real(c_float) :: moist(0:nx+1,1:nz+1,0:ny+1,n_moist),mut(0:nx+1,0:ny+1)
real(c_float) :: c1(nz+1),c2(nz+1),dnw(nz+1),p_top
real(c_float) :: p_hyd_w(0:nx+1,1:nz+1,0:ny+1),p_hyd(0:nx+1,1:nz+1,0:ny+1)
real(c_float) :: cqw(0:nx+1,1:nz+1,0:ny+1),cqu(0:nx+1,1:nz+1,0:ny+1),cqv(0:nx+1,1:nz+1,0:ny+1)
integer :: j_start,j_end,i_start,i_end,k_start,k_end,kte,i,j,k,n
real :: qtot
j_start=1
j_end=ny
i_start=1
i_end=nx
k_start=1
k_end=nz
kte=nz+1
call calc_cq(moist,cqu,cqv,cqw,n_moist,1,nx+1,1,ny+1,1,nz+1, &
             0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz+1)
''' + hydro + '''
end subroutine evaluate
end module original_moist
'''
    (directory / "original.f90").write_text(wrapper)
    compiler = os.environ.get("FC") or shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    command = [compiler, "-shared", "-fPIC", "-O0", "-ffp-contract=off",
               "-fno-fast-math", "-ffree-line-length-none", "-finit-real=zero",
               "-finit-integer=0", "-finit-logical=false", "original.f90", "-o", "original.so"]
    subprocess.run(command, cwd=directory, check=True, capture_output=True, timeout=60)
    metadata = {"source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "calc_cq_sha256": hashlib.sha256(cq.encode()).hexdigest(),
                "phy_prep_loop_sha256": hashlib.sha256(hydro.encode()).hexdigest(),
                "compile": command, "unchanged_extracted_bodies": True,
                "PARAM_FIRST_SCALAR": 2, "kind": "REAL4", "units": {"moist": "kg/kg", "pressure": "Pa", "cqw": "kg/kg"}}
    (directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    library = ctypes.CDLL(str(directory / "original.so"))
    function = library.evaluate
    function.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 9

    def evaluate(moist, mut, c1, c2, dnw, p_top):
        nz, ny, nx = moist[0].shape
        shape = (nx+2, nz+1, ny+2)
        q = np.zeros((*shape, len(moist)+1), np.float32, order="F")
        q[..., 0] = np.float32(1000)  # Unused WRF dummy must not load the air.
        for index, value in enumerate(moist, 1):
            q[1:nx+1, :nz, 1:ny+1, index] = np.asarray(value, np.float32).transpose(2, 0, 1)
        mass = np.zeros((nx+2, ny+2), np.float32, order="F")
        mass[1:nx+1, 1:ny+1] = np.asarray(mut, np.float32).T
        vector = lambda value: np.asfortranarray(np.r_[np.asarray(value, np.float32), np.float32(0)])
        arrays = [q, mass, vector(c1), vector(c2), vector(dnw), np.array(p_top, np.float32)]
        outputs = [np.zeros(shape, np.float32, order="F") for _ in range(3)]
        function(nx, ny, nz, len(moist)+1, *(a.ctypes.data for a in (*arrays, *outputs)))
        faces, pressure, cqw = (a[1:nx+1, :, 1:ny+1].transpose(1, 2, 0).copy() for a in outputs)
        return pressure[:nz], faces, cqw

    evaluate.metadata = metadata
    return evaluate
