"""CPU-only lowering/Triton/native static prescreen for the BATS radiation branch.

No CUDA context, device query, executable load or GPU timing. The one-call
pointwise interface is unchanged; full coupled GPU/E41 qualification is separate.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

TREE,OUT=map(Path,sys.argv[1:3])
opt_rad=int(sys.argv[3]) if len(sys.argv)>3 else 3
sys.path.insert(0,str(TREE/"src"))
os.environ["GPUWRF_NOAHMP_NATIVE_REAL"]="1"
import jax
import numpy as np
from gpuwrf.backends import pallas as facade
from jax._src.pallas.triton import lowering
from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
from gpuwrf.kernels.phys_noahmp_radiation import radiation_native
from gpuwrf.io.noahmp_options import read_bats_parameters

assert jax.devices()[0].platform=="cpu"
OUT.mkdir(parents=True,exist_ok=True)
fixture=TREE/"tests/v025/b_noahmp/test_snow_wrf_order_oracle.py"
spec=importlib.util.spec_from_file_location("u2_native_columns",fixture)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cols=json.loads((fixture.parent/"fixtures/swiss_snow_wrf_order.json").read_text())["columns"][:4]
c=module._build(cols)
phen=noahmp_phenology_table(c["land"],c["forcing"],c["static"])
facade.cuda_native=lambda:True  # trace native Pallas; actual backend remains CPU
bats=read_bats_parameters()
geometry=None
if opt_rad==1:
    import jax.numpy as jnp
    p=c['static'].parameters;ivg=c['static'].ivgtyp-1
    geometry=tuple(jnp.asarray(getattr(p,key),jnp.float32)[ivg] for key in ('rc','hvt','hvb'))
call=lambda land,forcing,params,phen:radiation_native(land,forcing,None,phen,params,c["dt"],opt_alb=1,bats=bats,
    **({'opt_rad':1,'geometry':geometry} if opt_rad==1 else {}))
t0=time.monotonic()
jp=jax.make_jaxpr(call)(c["land"],c["forcing"],c["rp"],phen).jaxpr
eqns=[eqn for eqn in jp.eqns if eqn.primitive.name=="pallas_call"]
assert len(eqns)==1,"BATS must retain one Pallas radiation call"
params=eqns[0].params
def double_ops(jp):
    if hasattr(jp,'jaxpr'):jp=jp.jaxpr
    count=0
    for equation in jp.eqns:
        count+=sum(getattr(getattr(v,'aval',None),'dtype',None)==np.dtype('float64') for v in equation.outvars)
        for value in equation.params.values():
            if hasattr(value,'jaxpr') or hasattr(value,'eqns'):count+=double_ops(value)
    return count
source_f64=double_ops(params['jaxpr'])
assert source_f64==0,'Native source jaxpr must be REAL4'
lowered=lowering.lower_jaxpr_to_triton_module(params["jaxpr"],params["grid_mapping"],"cuda",120)
ir=str(lowered.module)
path=OUT/"bats.ttir";path.write_text(ir)
report=dict(platform="cpu",gpu_executed=False,target="cuda/sm120",native_calls=1,num_warps=4,block=128,
    opt_rad=opt_rad,
    source_jaxpr_f64_outputs=source_f64,
    lowering_s=time.monotonic()-t0,ir_bytes=len(ir),source_tree=str(TREE),
    source_commit=subprocess.check_output(["git","-C",str(TREE),"rev-parse","HEAD"],text=True).strip(),
    source_sha256=hashlib.sha256((TREE/"src/gpuwrf/kernels/phys_noahmp_radiation.py").read_bytes()).hexdigest())
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
import triton
from triton.backends.compiler import GPUTarget
t0=time.monotonic()
kernel=triton.compile(str(path),target=GPUTarget("cuda",120,32))
report["compile_s"]=time.monotonic()-t0
ptx=kernel.asm["ptx"]
report["ptx_f64_ops"]=len(re.findall(r"^\s*(?:@\S+\s+)?[A-Za-z][^;\n]*\.f64\b",ptx,re.M))
report['libdevice_trig_f64_disclosure']=('CUDA tanf/cosf generic slow range reduction emits fp64 conversion/multiply. '
    'All source jaxpr operations are REAL4. Canopy trig arguments are bounded by solar cosine .01..1 and atan; '
    'slow reduction is unreachable for these finite inputs. GPU runtime and full executable qualification pending.'
    if opt_rad==1 and report['ptx_f64_ops'] else None)
assert report["ptx_f64_ops"]==0 or opt_rad==1
cubin=OUT/"bats.cubin";cubin.write_bytes(kernel.asm["cubin"])
report["cubin_sha256"]=hashlib.sha256(cubin.read_bytes()).hexdigest()
scan=subprocess.run([sys.executable,str(TREE/"scripts/e41/cubin_scan.py"),str(cubin),"unused","bats"],cwd=OUT,
                    capture_output=True,text=True,timeout=60)
(OUT/"scan.log").write_text(scan.stdout+scan.stderr)
report["native_scanner_rc"]=scan.returncode
report["scope"]="component native static prescreen only; whole coupled GPU executable/HLO gate pending"
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
assert scan.returncode==0,"native component static gate failed; inspect scan.log"
