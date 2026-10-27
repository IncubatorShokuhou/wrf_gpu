"""Original WRF REAL rain-init constants at three recompiled av_r values."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert os.environ.get("JAX_PLATFORMS")=="cpu"
root=Path(__file__).resolve().parents[2]
source=root/"proofs/thompson/wrf_constants/wrf_constants.F90"
compiler="<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
flags=["-O2","-ftree-vectorize","-funroll-loops","-ffree-form","-ffree-line-length-none"]
assert not args.out.exists()
args.out.mkdir(parents=True)
reference={}
names=("T1_QR_QC","T2_QR_QI","T2_QR_EV")
for value in (2427.,4854.,9708.):
    with tempfile.TemporaryDirectory(prefix="rain_init_",dir=args.out) as temporary:
        folder=Path(temporary)
        text=source.read_text()
        assert text.count("av_r = 4854.0")==1
        changed=text.replace("av_r = 4854.0",f"av_r = {value:.1f}")
        path=folder/"constants.f90";path.write_text(changed)
        subprocess.run([compiler,*flags,str(path),"-o",str(folder/"constants")],cwd=folder,check=True)
        result=subprocess.check_output([str(folder/"constants")],cwd=folder,text=True)
        rows={row.split()[0]:row.split()[1:] for row in result.splitlines() if row.split()}
        reference[str(value)]={name:{"bits":int(rows[name][0]),"value":float(rows[name][1])} for name in names}
        (args.out/(str(int(value))+".txt")).write_text(result)
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.thompson_rain_parameters import rain_parameters
assert jax.devices()[0].platform=="cpu"
calls=[]


def program(value):
    calls.append(1)
    return rain_parameters(value)


compiled=jax.jit(program)
rows={}
for value in (2427.,4854.,9708.):
    expected=np.array([reference[str(value)][name]["bits"] for name in names],np.uint32).view(np.float32)
    eager=np.asarray(rain_parameters(jnp.float32(value)))
    actual=np.asarray(compiled(jnp.float32(value)))
    assert np.isfinite(actual).all() and np.isfinite(eager).all()
    rows[str(value)]=dict(reference_bits=expected.view(np.uint32).tolist(),eager_bits=eager.view(np.uint32).tolist(),
                         compiled_bits=actual.view(np.uint32).tolist(),eager_byte=expected.tobytes()==eager.tobytes(),
                         compiled_ulp=np.abs(actual.view(np.uint32).astype(np.int64)-expected.view(np.uint32).astype(np.int64)).tolist())
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),compiler=compiler,flags=flags,
            platform="cpu",gpu_executed=False,reference=reference,results=rows,traces=len(calls),
            scope="init-coefficient oracle only; whole-column default/original-WRF gates remain separate")
(args.out/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
assert len(calls)==1
assert all(row["eager_byte"] for row in rows.values()),rows
