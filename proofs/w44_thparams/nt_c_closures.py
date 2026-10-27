"""Measure Nt scalar preparation and locate default cloud-moment rounding.

Compare exact WRF table/index controls before interpreting a changed final
field. No GPU or trajectory claim; all inputs are the frozen real columns.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics.thompson_nt_c import cloud_parameters, host_tables

parser=argparse.ArgumentParser()
parser.add_argument("--fixture",type=Path,required=True)
parser.add_argument("--selection",type=Path,required=True)
parser.add_argument("--index-reference",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert jax.devices()[0].platform=="cpu"
args.out.mkdir(parents=True,exist_ok=True)
expected=np.array([12,65,tc.WRF_REAL_CONSTANTS["LAMC_PREFIX"],tc.WRF_REAL_CONSTANTS["DC_G_PREFIX"],
                   tc.WRF_REAL_CONSTANTS["PNR_WAU_DEN"],tc.WRF_REAL_CONSTANTS["CCG5_NU12"],
                   tc.WRF_REAL_CONSTANTS["OCG2_NU12"]],np.float32)
compiled=jax.jit(cloud_parameters)
eager=np.asarray(cloud_parameters(jnp.float32(100.e6)))
default=np.asarray(compiled(jnp.float32(100.e6)))
assert np.array_equal(eager.view(np.uint32),expected.view(np.uint32))
indices=[]
for row in json.loads(args.index_reference.read_text()):
    values=np.asarray(compiled(jnp.float32(row["nt_c"])))
    passed=int(values[0])==row["nu_c"] and int(values[1])+1==row["idx_n"]
    indices.append(dict(nt_c=row["nt_c"],nu=int(values[0]),idx=int(values[1])+1,passed=passed))
assert all(row["passed"] for row in indices)
z=dict(np.load(args.fixture,allow_pickle=False))
idx=[z["names"].tolist().index(n) for n in json.loads(args.selection.read_text())["selected_columns"]]
f=lambda name:jnp.asarray(z["in_"+name][idx],jnp.float32)
temp,p,qv=f("th")*f("pii"),f("p"),f("qv")
qc,rho=f("qc"),tc.density_from_pressure_temperature(p,temp,qv)


def variant(qc,rho,nt,pack,which):
    env=dict(vars(tc))
    if which in ("all","number"):
        env["NT_C"]=nt
    if which in ("all","shape"):
        env["NU_C_MP8"]=pack[0]
    if which in ("all","slope"):
        env["WRF_REAL_CONSTANTS"]={**tc.WRF_REAL_CONSTANTS,"LAMC_PREFIX":pack[2]}
    original=tc._wrf
    env["_wrf"]=FunctionType(original.__code__,env,original.__name__,original.__defaults__)
    original=tc._cloud_distribution
    fn=FunctionType(original.__code__,env,original.__name__,original.__defaults__)
    return fn(qc,rho)


baseline=[np.asarray(a) for a in jax.jit(tc._cloud_distribution)(qc,rho)]
rows={}
for which in ("all","number","shape","slope"):
    fn=jax.jit(lambda qc,rho,nt,pack:variant(qc,rho,nt,pack,which))
    out=[np.asarray(a) for a in fn(qc,rho,jnp.float32(100.e6),jnp.asarray(default))]
    assert all(np.isfinite(a).all() for a in out)
    rows[which]=[dict(field=name,changed=int(np.count_nonzero(a!=b)),maxabs=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
                 for name,a,b in zip(("rc","lamc","xdc","mvd","active"),baseline,out,strict=True)]
    (args.out/(which+".hlo")).write_text(fn.lower(qc,rho,jnp.float32(100.e6),jnp.asarray(default)).compile().as_text())
receipt=dict(platform="cpu",gpu_executed=False,index_controls=indices,
             eager_prefix_bits=eager.view(np.uint32).tolist(),expected_bits=expected.view(np.uint32).tolist(),
             compiled_prefix_bits=default.view(np.uint32).tolist(),compiled_prefix_delta=(default-expected).tolist(),
             moments=rows, fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
             index_reference_sha256=hashlib.sha256(args.index_reference.read_bytes()).hexdigest(),
             old_mvd_numerator=float(3.+12.+.672),new_REAL_mvd_numerator=float(np.float32(np.float32(3)+np.float32(12))+np.float32(.672)))
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
