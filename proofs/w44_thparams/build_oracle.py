"""Private recompile of the original WRF Thompson module with selected constants.

The sole scientific change is each requested declaration or source literal. Link to
the pristine WRF library and use the established full mp_gt_driver caller.
Never modify original WRF sources, objects or tables. No GPU involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--tables",type=Path,required=True)
parser.add_argument("--fixture",type=Path,required=True)
parser.add_argument("--selection",type=Path,required=True)
parser.add_argument("--values",default="{}")
args=parser.parse_args()
wrf=Path("<USER_HOME>/src/wrf_pristine/WRF")
buildenv=Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build")
compiler=buildenv/"bin/x86_64-conda-linux-gnu-gfortran"
flags=["-w","-ffree-form","-ffree-line-length-none","-fconvert=big-endian","-frecord-marker=4",
       "-fallow-argument-mismatch","-fallow-invalid-boz","-O2","-ftree-vectorize","-funroll-loops"]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
values=json.loads(args.values)
assert set(values)<=set(("nt_c","tno","ato","av_c","av_i","av_r","autoconversion_threshold","autoconversion_rate"))
assert not args.out.exists(), "refuse to overwrite an oracle arm"
args.out.mkdir(parents=True)
src=(wrf/"phys/module_mp_thompson.f90").read_text()
original=src
for name,value in values.items():
    assert np.isfinite(value) and (value>=0 if name=="autoconversion_rate" else value>0)
    if name=="autoconversion_threshold":
        assert np.float32(value)>=np.float32(1.e-12)
        pattern=r"(if\s*\(rc\(k\)\.gt\.\s*)0\.01e-3(\)\s*then)"
        src,count=re.subn(pattern,lambda match:match[1]+f"{value:.16e}"+match[2],src,flags=re.I)
    elif name=="autoconversion_rate":
        pattern=r"(zeta\s*=\s*)0\.027(\s*\*\s*rc\(k\)\s*\*\s*zeta1)"
        src,count=re.subn(pattern,lambda match:match[1]+f"{value:.16e}"+match[2],src,flags=re.I)
    else:
        pattern=rf"(REAL, PARAMETER, PRIVATE:: {name} = )[^\n]+"
        src,count=re.subn(pattern,lambda match:match[1]+f"{value:.16e}",src,flags=re.I)
    assert count==1,(name,count)
(args.out/"module_mp_thompson.f90").write_text(src)
caller=Path(__file__).resolve().parents[1]/"thompson/ice_process_oracle/thompson_column_driver.F90"
(args.out/"thompson_column_driver.F90").write_bytes(caller.read_bytes())
tables={name:args.tables/name for name in ("qr_acr_qg_V4.dat","qr_acr_qsV2.dat","freezeH2O.dat")}
tables["CCN_ACTIVATE.BIN"]=wrf/"run/CCN_ACTIVATE.BIN"
for name,path in tables.items():
    assert path.is_file(),path
    (args.out/name).symlink_to(path.resolve())
commands=[]
started=time.monotonic()


def run(command):
    commands.append([str(x) for x in command])
    subprocess.run(commands[-1],cwd=args.out,check=True)


inc=["-I"+str(wrf/d) for d in ("phys","frame","share","main","inc")]
run([compiler,*flags,*inc,"-c","module_mp_thompson.f90","-o","module_mp_thompson.o"])
run([compiler,*flags,"-I.",*inc,"-c","thompson_column_driver.F90","-o","driver.o"])
run([compiler,*flags,"driver.o","module_mp_thompson.o",wrf/"main/libwrflib.a",
     wrf/"external/fftpack/fftpack5/libfftpack.a",wrf/"external/io_grib1/libio_grib1.a",
     wrf/"external/io_grib_share/libio_grib_share.a",wrf/"external/io_int/libwrfio_int.a",
     "-L"+str(wrf/"external/esmf_time_f90"),"-lesmf_time",wrf/"frame/module_internal_header_util.o",
     wrf/"frame/pack_utils.o","-L"+str(wrf/"external/io_netcdf"),"-lwrfio_nf",
     "-L"+str(buildenv/"lib"),"-lnetcdff","-lnetcdf","-Wl,-rpath,"+str(buildenv/"lib"),"-o","oracle"])
z=dict(np.load(args.fixture,allow_pickle=False))
selection=json.loads(args.selection.read_text())
idx=np.array([z["names"].tolist().index(n) for n in selection["selected_columns"]])
ncol,nlev=len(idx),z["in_qv"].shape[1]
assert np.all(z["dt"][idx]==18.)
with (args.out/"input.bin").open("wb") as stream:
    stream.write(np.array([ncol,nlev],">i4").tobytes())
    stream.write(np.array([18.],">f4").tobytes())
    for name in ("qv","qc","qr","qi","qs","qg","ni","nr","th","pii","p","w","dz8w"):
        a=np.asarray(z["in_"+name][idx].T,">f4")
        assert np.isfinite(a).all(),name
        stream.write(a.tobytes())
run([args.out/"oracle","input.bin","output.bin"])
raw=np.fromfile(args.out/"output.bin",">f4")
out,offset={},0
for name in ("qv","qc","qr","qi","qs","qg","ni","nr","th"):
    out[name]=raw[offset:offset+ncol*nlev].reshape(nlev,ncol).T.astype(np.float32)
    offset+=ncol*nlev
for name in ("rainncv","snowncv","graupelncv","sr"):
    out[name]=raw[offset:offset+ncol].astype(np.float32)
    offset+=ncol
assert offset==raw.size and np.isfinite(raw).all()
np.savez_compressed(args.out/"reference.npz",**out)
receipt=dict(values=values,platform="cpu",gpu_executed=False,original_F_sha256=sha(wrf/"phys/module_mp_thompson.F"),
             original_f90_sha256=hashlib.sha256(original.encode()).hexdigest(),
             changed_f90_sha256=sha(args.out/"module_mp_thompson.f90"),caller_sha256=sha(caller),
             object_sha256=sha(args.out/"module_mp_thompson.o"),executable_sha256=sha(args.out/"oracle"),
             fixture_sha256=sha(args.fixture),selection=selection["selected_columns"],
             tables={name:dict(path=str(path),sha256=sha(path)) for name,path in tables.items()},
             commands=commands,wall_s=time.monotonic()-started,output_sha256=sha(args.out/"output.bin"))
if not values:
    receipt["original_fixture_byte_exact"]={name:out[name].tobytes()==z["wrf_"+name][idx].tobytes()
                                             for name in out if "wrf_"+name in z}
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
if not values:
    assert all(receipt["original_fixture_byte_exact"].values()), receipt["original_fixture_byte_exact"]
