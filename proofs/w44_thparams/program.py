"""Native Thompson CPU lowering/default-byte/changed-value proof.

Run this fixed tool through a stable import-src alias for raw OFF HLO identity.
The input is the frozen real mixed-phase WRF-column corpus (E154).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--import-src", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--action", choices=("lower", "evaluate"), required=True)
parser.add_argument("--values", help="JSON parameter mapping; absent means OFF")
parser.add_argument("--variants", help="JSON named value overrides for the CPU evaluate arm")
parser.add_argument("--ncol", type=int, default=24)
args = parser.parse_args()
sys.path.insert(0, str(args.import_src))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
from gpuwrf.kernels.phys_thompson_full import full_column
from gpuwrf.physics import thompson_column as tc

assert jax.devices()[0].platform == "cpu"
source_root=Path(tc.__file__).resolve().parents[3]
pin,src_tree=subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD","HEAD:src"],text=True).splitlines()
assert not subprocess.check_output(["git","-C",str(source_root),"status","--porcelain","--untracked-files=no"])
z = dict(np.load(args.fixture, allow_pickle=False))
assert all(np.isfinite(v).all() for k, v in z.items() if k.startswith(("in_", "wrf_")))
candidates = np.flatnonzero(z["dt"] == 18.)
# The corpus is grouped by source/process; its first rows omit graupel. Seed
# each hydrometeor maximum, then span the remaining source times/domains.
seeds = [int(candidates[np.argmax(np.max(z["in_"+v][candidates], axis=1))])
         for v in ("qc","qr","qi","qs","qg")]
spaced = candidates[np.linspace(0,len(candidates)-1,args.ncol,dtype=int)].tolist()
idx = np.array(list(dict.fromkeys(seeds+spaced))[:args.ncol])
f = lambda name: jnp.asarray(z["in_"+name][idx], jnp.float32)
temp, p, qv = f("th")*f("pii"), f("p"), f("qv")
zero = jnp.zeros_like(qv)
state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"),
                              Ni=f("ni"), Nr=f("nr"), Ns=zero, Ng=zero, T=temp, p=p,
                              rho=tc.density_from_pressure_temperature(p,temp,qv), dz=f("dz8w"), w=f("w"))
traces = []


def program(state, parameters):
    traces.append(1)
    kw = {} if parameters is None else {"parameters": parameters}
    return full_column(state, 18., interpret=args.action == "evaluate", **kw)


fn = jax.jit(program)
receipt = dict(platform="cpu", gpu_executed=False, fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
               pin=pin,src_tree=src_tree,
               source=str(Path(tc.__file__).resolve()), ncol=len(idx), nz=state.qv.shape[-1], dt=18.,
               selected_columns=z["names"][idx].tolist(), input_nonzero={v:int(np.count_nonzero(z["in_"+v][idx]))
                                                                  for v in ("qc","qr","qi","qs","qg")})
assert all(receipt["input_nonzero"].values())
args.out.mkdir(parents=True, exist_ok=True)
if args.action == "lower":
    # E115: trace the native facade and lower for CUDA without creating a CUDA
    # backend. jax.devices() remains CPU; no executable is compiled or run.
    from gpuwrf.backends import pallas as facade
    facade.cuda_native = lambda: True
    jax.default_backend = lambda: "gpu"
    params = None if args.values is None else {k:jnp.float32(v) for k,v in json.loads(args.values).items()}
    trace = fn.trace(state, params)
    lowered = trace.lower(lowering_platforms=("cuda",))
    hlo = lowered.compiler_ir("hlo").as_hlo_text()
    jp = str(jax.make_jaxpr(program)(state, params))
    canon = re.sub(r"name_and_src_info=[^\]\n]*", "", jp)
    canon = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", canon)
    receipt.update(raw_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),raw_hlo_bytes=len(hlo.encode()),
                   canonical_jaxpr_sha256=hashlib.sha256(canon.encode()).hexdigest(),
                   native_custom_calls=hlo.count("custom-call("), parameters=args.values)
    (args.out/"raw.hlo").write_text(hlo)
    (args.out/"canonical.jaxpr").write_text(canon)
else:
    from gpuwrf.physics.thompson_parameters import PARAMETERS, validate_parameters
    defaults = ({k:v[0] for k,v in PARAMETERS.items()} if args.values is None else json.loads(args.values))
    outputs = []
    arms = [("off", None), ("default", defaults)]
    if args.variants is None:
        arms += [(k,{**defaults, k:v*2}) for k,v in defaults.items()]
    else:
        variants=json.loads(args.variants)
        assert not set(variants)&{"off","default"}
        arms += [(label,{**defaults,**values}) for label,values in variants.items()]
    for label, values in arms:
        out = fn(state, None if values is None else validate_parameters(values))
        arrays = [np.asarray(a).copy() for a in jax.tree.leaves(out)]
        assert all(np.isfinite(a).all() for a in arrays), label
        outputs.append(arrays)
        np.savez_compressed(args.out/(label+".npz"), **{str(i):a for i,a in enumerate(arrays)})
    receipt["output_leaves"] = len(outputs[0])
    from roundoff import compare
    receipt["default_roundoff"] = compare(outputs[0], outputs[1], nt_c="nt_c" in defaults)
    receipt["default_byte_exact"] = all(a.tobytes()==b.tobytes() for a,b in zip(outputs[0],outputs[1],strict=True))
    receipt["default_differences"] = [dict(leaf=i, cells=int(np.count_nonzero(a!=b)),maxabs=float(np.max(np.abs(a-b))))
                                      for i,(a,b) in enumerate(zip(outputs[0],outputs[1],strict=True)) if a.tobytes()!=b.tobytes()]
    receipt["changed_leaves"] = {name:sum(a.tobytes()!=b.tobytes() for a,b in zip(outputs[1],values,strict=True))
                                 for (name,_),values in zip(arms[2:],outputs[2:],strict=True)}
    receipt["traces"] = len(traces)
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
if args.action == "lower":
    assert receipt["native_custom_calls"] == 1, receipt["native_custom_calls"]
if args.action == "evaluate":
    assert receipt["default_roundoff"]["within_roundoff"], receipt["default_roundoff"]
    assert len(traces) == 2, len(traces)
    assert all(receipt["changed_leaves"].values()), receipt["changed_leaves"]
