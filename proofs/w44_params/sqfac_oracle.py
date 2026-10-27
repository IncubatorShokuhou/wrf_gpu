"""Original WRF mym_predict/tridiag2 bodies with SQFAC supplied by the caller.

The default arm also links the original WRF objects; require byte equality to
that arm before using the extracted routines for a changed SQFAC. No pristine
source/object is edited. The wrapper mirrors the established BP116 caller.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRAPPER = """
subroutine predict(n,dt,ust,pmz,factor,dz,rho,el,dfq,pdk_in,pdt_in,pdq_in,pdc_in,qke_in,qsq_in,saw,out) bind(C)
use iso_c_binding
use @MODULE@,only:mym_predict @IMPORT@
implicit none
integer(c_int),value::n
real(c_float),value::dt,ust,pmz,factor
real(c_float),intent(in)::dz(n),rho(n),el(n),dfq(n),pdk_in(n),pdt_in(n),pdq_in(n),pdc_in(n),qke_in(n),qsq_in(n),saw(n+1)
real(c_float),intent(out)::out(n,4)
real(c_float)::pdk(n),pdt(n),pdq(n),pdc(n),qke(n),tsq(n),qsq(n),cov(n),saw_copy(n+1),sawqke(n+1),qwt(n),qdiss(n)
pdk=pdk_in;pdt=pdt_in;pdq=pdq_in;pdc=pdc_in;qke=qke_in;qsq=qsq_in;tsq=0.;cov=0.;saw_copy=saw;sawqke=0.
@SET@
call mym_predict(1,n,2.6_c_float,dt,dz,ust,0._c_float,0._c_float,pmz,1._c_float,el,dfq,rho, &
 pdk,pdt,pdq,pdc,qke,tsq,qsq,cov,saw_copy,sawqke,0,qwt,qdiss,0)
out(:,1)=qke;out(:,2)=qsq;out(:,3)=tsq;out(:,4)=cov
end subroutine
"""


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    phys = WRF / "phys"
    source = (phys / "module_bl_mynnedmf.F").read_text()
    routines = {name: re.search(r"(?ims)^\s*subroutine\s+" + name + r"\b.*?^\s*end subroutine\s+" + name + r"\b", source).group(0)
                for name in ("mym_predict", "tridiag2")}
    constants = {name: re.search(r"(?i)\b" + name + r"\s*=\s*([0-9.eE+\-]+)", source).group(1)
                 for name in ("b1", "b2", "qkemin", "sqfac")}
    assert constants == {"b1": "24.0", "b2": "15.0", "qkemin": "1.e-5", "sqfac": "3.0"}
    header = ("module w44_predict\nuse module_bl_mynnedmf_common\n"
              "implicit none\nreal(kind_phys),parameter::" +
              ",".join(name + "=" + constants[name] for name in ("b1", "b2", "qkemin")) +
              "\nreal(kind_phys)::sqfac\ncontains\n")
    extracted = header + "\n".join(routines.values()) + "\nend module\n"
    (directory / "predict.F90").write_text(extracted)
    commands, paths = [], {}
    compiler = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    for label in ("original", "runtime"):
        wrapper = WRAPPER.replace("@MODULE@", "module_bl_mynnedmf" if label == "original" else "w44_predict")
        wrapper = wrapper.replace("@IMPORT@", "" if label == "original" else ",sqfac")
        wrapper = wrapper.replace("@SET@", "" if label == "original" else "sqfac=factor")
        path = directory / (label + ".f90")
        path.write_text(wrapper)
        core = [str(phys / "module_bl_mynnedmf.o")] if label == "original" else [str(directory / "predict.F90")]
        output = directory / (label + ".so")
        command = [compiler, "-cpp", "-O2", "-fPIC", "-shared", "-ffree-line-length-none", "-ffp-contract=off",
                   "-I" + str(phys), "-I" + str(directory), "-J" + str(directory), *core, str(path),
                   str(phys / "module_bl_mynnedmf_common.o"), "-o", str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, cwd=directory)
        commands.append(command)
        paths[label] = output
    (directory / "manifest.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "routine_sha256": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in routines.items()},
        "constants": constants, "wrapper_sha256": hashlib.sha256(WRAPPER.encode()).hexdigest(),
        "original_objects": {str(phys / name): hashlib.sha256((phys / name).read_bytes()).hexdigest()
                             for name in ("module_bl_mynnedmf.o", "module_bl_mynnedmf_common.o")},
        "caller": {"closure": 2.6, "edmf_tke": 0, "tke_budget": 0, "sawqke": 0},
        "commands": commands}, indent=2))
    return paths


def evaluate(library, state, flux, turb, saw, pmz, dt, factor):
    lib = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.predict.argtypes = [ctypes.c_int] + [ctypes.c_float] * 4 + [pointer] * 12
    ncol, nz = state.u.shape
    output = np.empty((ncol, 4, nz), np.float32)
    values = [state.dz, state.rho, turb["el"], turb["dfq"], turb["pdk"], turb["pdt"],
              turb["pdq"], turb["pdc"], 2 * state.tke, state.qsq, saw]
    for col in range(ncol):
        arrays = [np.ascontiguousarray(np.asarray(value)[col], np.float32) for value in values]
        assert all(np.isfinite(value).all() for value in arrays)
        lib.predict(nz, dt, float(flux.ustar[col]), float(pmz[col]), factor, *arrays, output[col])
    assert np.isfinite(output).all()
    return output
