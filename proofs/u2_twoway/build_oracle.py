"""Build feedback savepoints from VERBATIM WRF copy_fcn/sm121/smdsm.

Run on CPU with a lane-owned work directory. No WRF executable or reference file
is modified. The standalone configuration stub supplies only nl_get_spec_zone;
the bounds/staggering match the serial Registry feedback caller (REAL*4).
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/share/interp_fcn.F")
FLAGS = ["-shared", "-fPIC", "-ffree-form", "-ffree-line-length-none", "-O2",
         "-march=nocona", "-ffp-contract=off", "-fcheck=bounds"]


def build(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_text()
    routines = {}
    for name in ("copy_fcn", "copy_fcnm", "copy_fcni", "sm121", "smdsm"):
        match = re.search(rf"^   SUBROUTINE {name} \(.*?^   END SUBROUTINE {name}[^\n]*",
                          raw, flags=re.M | re.S | re.I)
        assert match, name
        routines[name] = match.group(0)
    stub = """module module_wrf_error
      end module
      module module_configure
      integer :: zone = 1
      contains
      subroutine nl_get_spec_zone(id,value)
      integer :: id,value
      value=zone
      end subroutine
      end module
      subroutine set_zone(value)
      use module_configure
      integer :: value
      zone=value
      end subroutine
"""
    code = work / "feedback_oracle.f90"
    code.write_text(stub + "\n" + "\n\n".join(routines.values()) + "\n")
    lib = work / "feedback_oracle.so"
    subprocess.run([FC, *FLAGS, "-o", str(lib), str(code)], cwd=work, check=True)
    manifest = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "routines": {k: hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},
                "compiler": FC, "flags": FLAGS, "precision": "WRF REAL binary32",
                "scope": "serial feedback operators; NOT a coupled forecast qualification"}
    return ct.CDLL(str(lib)), manifest


def call(lib, parent, child, *, ratio=3, ipos=3, jpos=3, stagger="", zone=1, smooth=0, method="copy_fcn"):
    """Run the pristine operators, preserving parent cells outside WRF bounds."""
    is2d = parent.ndim == 2
    dtype = np.int32 if method=="copy_fcni" else np.float32
    p = np.asarray(parent if not is2d else parent[None], dtype=dtype)
    c = np.asarray(child if not is2d else child[None], dtype=dtype)
    nz, pny, pnx = p.shape
    _, cny, cnx = c.shape
    # WRF memory is (i,k,j); logical domain end is one beyond mass-grid extent.
    pf = np.asfortranarray(p.transpose(2,0,1))
    cf = np.asfortranarray(c.transpose(2,0,1))
    pmx, pmy = pnx - (stagger == "U"), pny - (stagger == "V")
    cmx, cmy = cnx - (stagger == "U"), cny - (stagger == "V")
    pdims = [1,pmx+1,1,nz+1,1,pmy+1, 1,pnx,1,nz,1,pny, 1,pnx,1,nz,1,pny]
    cdims = [1,cmx+1,1,nz+1,1,cmy+1, 1,cnx,1,nz,1,cny, 1,cnx,1,nz,1,cny]
    ints = lambda vs: [ct.byref(ct.c_int(int(v))) for v in vs]
    ptr = lambda arr: arr.ctypes.data_as(ct.POINTER(ct.c_int if dtype==np.int32 else ct.c_float))
    mask = np.ones((cnx,cny), dtype=np.int32, order="F")
    lib.set_zone_(ct.byref(ct.c_int(zone)))
    getattr(lib,method+"_")(ptr(pf), *ints(pdims), ptr(cf), *ints(cdims), *ints([0]),
                  mask.ctypes.data_as(ct.POINTER(ct.c_int)),
                  *ints([stagger=="U",stagger=="V",ipos,jpos,ratio,ratio]))
    if smooth:
        routine = lib.sm121_ if smooth == 1 else lib.smdsm_
        routine(ptr(pf), *ints(pdims), *ints([stagger=="U",stagger=="V"]),
                *ints(cdims), *ints([ratio,ratio,ipos,jpos]))
    out = pf.transpose(1,2,0).copy()
    return out[0] if is2d else out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    lib, manifest = build(args.work.resolve())
    payload = {}
    cases = []
    rng = np.random.default_rng(20261013)
    for ratio, zone in ((3,1),(3,2),(5,1)):
        for stagger in ("", "U", "V"):
            for smooth in (0,1,2):
                idx = str(len(cases))
                pshape = (2,20+(stagger=="V"),22+(stagger=="U"))
                cshape = (2,10*ratio+(stagger=="V"),12*ratio+(stagger=="U"))
                p = rng.normal(20,5,pshape).astype(np.float32)
                c = rng.normal(25,9,cshape).astype(np.float32)
                # Three feedback epochs, each with a distinct nonuniform child.
                payload[f"p{idx}"] = p
                payload[f"c{idx}"] = c
                opts = dict(ratio=ratio,zone=zone,stagger=stagger,smooth=smooth)
                for step in range(3):
                    p = call(lib,p,c+np.float32(step)*np.float32(0.125),**opts)
                    payload[f"o{idx}_{step}"] = p
                cases.append(opts)
    # Real PROD wrfinput operands, cropped to a two-domain operator geometry.
    # PROD is WN2: operator diagnostics only, never a release validation claim.
    from gpuwrf.io.netcdf_lock import Dataset
    case = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
    with Dataset(case / "wrfinput_d01") as pd, Dataset(case / "wrfinput_d02") as cd:
        fields = {"U":"U", "V":"V", "W":"", "T":"", "PH":"", "PHB":"",
                  "MU":"", "MUB":"", "QVAPOR":"", "QCLOUD":"", "QRAIN":"", "QICE":""}
        census = {}
        for name, stagger in fields.items():
            p = np.asarray(pd[name][0],dtype=np.float32)[..., :20+(stagger=="V"), :22+(stagger=="U")]
            c = np.asarray(cd[name][0],dtype=np.float32)[..., :30+(stagger=="V"), :36+(stagger=="U")]
            census[name] = {"parent_nonzero":int(np.count_nonzero(p)),"child_nonzero":int(np.count_nonzero(c))}
            for smooth in (0,1,2):
                idx = str(len(cases))
                payload[f"p{idx}"] = p
                payload[f"c{idx}"] = c
                opts = dict(ratio=3,zone=1,stagger=stagger,smooth=smooth)
                out = p
                for step in range(3):
                    out = call(lib,out,c,**opts)
                    payload[f"o{idx}_{step}"] = out
                cases.append({**opts,"field":name})
    manifest["real_input_census"] = census
    manifest["real_case"] = str(case)
    manifest["cases"] = cases
    manifest["epochs"] = 3
    manifest["input_census"] = "all donors nonzero and spatially nonuniform; all 3 staggerings and 0/1/2 smooth options"
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture, **payload)
    manifest["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    args.fixture.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
