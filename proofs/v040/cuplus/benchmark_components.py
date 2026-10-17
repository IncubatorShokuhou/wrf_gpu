"""Small CPU compile/workspace diagnostic; no GPU/forecast speed claim."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import cumulus_kfcup_support as cup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    assert os.environ["JAX_PLATFORMS"]=="cpu"
    assert jax.devices()[0].platform=="cpu"
    fixture=json.loads(Path(__file__).with_name("fixtures.json").read_text())
    cloud=next(c for c in fixture["cases"] if c["id"]=="cloud_shallow")
    radiation=next(c for c in fixture["cases"] if c["id"]=="radiation_s1_i1_f1")
    real=lambda x:jnp.asarray(x,jnp.float32)
    cloud_inputs=[real(cloud[k]) for k in ("qlg","qig","qv","temperature","height","pressure")]
    cloud_kw={k:real(cloud[k]) for k in ("wstar","wparcel","pblh","active_frac","mfup")}
    rad_inputs=[real(radiation[k]) for k in ("cldfra","qc","cldfra_cup")]
    rad_inputs += [jnp.asarray(radiation["cldfra1_flag"],jnp.int32)]
    rad_kw={k:real(radiation[k]) for k in ("ishall","cloud_bottom","cloud_top")}
    rad_kw.update(icloud=jnp.int32(radiation["icloud"]),shallowcu_forced_ra=radiation["shallowcu_forced_ra"])
    programs=(
        ("jfd9",jax.jit(partial(cup.cup_jfd,num_bins=9)),
         [real(x) for x in (1200.,-800.,.002,.8)],dict(th_bin_size=real(.1),r_bin_size=real(.0001))),
        ("cloud6",jax.jit(partial(cup.cup_cloud_fraction,ishall=1,cloud_bottom=1,cloud_top=3)),cloud_inputs,cloud_kw),
        ("radiation6",jax.jit(cup.cup_radiation_feedback),rad_inputs,rad_kw))
    rows=[]
    for name,function,inputs,kwargs in programs:
        start=time.perf_counter()
        lowered=function.lower(*inputs,**kwargs)
        executable=lowered.compile()
        compile_s=time.perf_counter()-start
        output=jax.block_until_ready(executable(*inputs,**kwargs))
        assert all(np.isfinite(np.asarray(v)).all() for v in output.values())
        samples=[]
        for _ in range(31):
            start=time.perf_counter_ns()
            jax.block_until_ready(executable(*inputs,**kwargs))
            samples.append((time.perf_counter_ns()-start)/1e6)
        memory=executable.memory_analysis()
        fields=("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes","generated_code_size_in_bytes")
        hlo=str(lowered.compiler_ir(dialect="stablehlo"))
        assert "f64" not in hlo
        rows.append(dict(program=name,compile_seconds=compile_s,cpu_warm_median_ms=float(np.median(samples)),
                         memory={k:int(getattr(memory,k)) for k in fields},stablehlo_f64_present=False))
    root=Path(__file__).resolve().parents[3]
    receipt=dict(platform="cpu",source_commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
                 cache="NONE (GPUWRF_JAX_CACHE=0)",scope="component CPU cost only; no GPU wall/VRAM/E41/launch count or forecast claim",
                 jax_version=jax.__version__,rows=rows)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
