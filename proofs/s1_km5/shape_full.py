"""Cheap shape preflight before a full SMS numerical/coupled arm."""
from pathlib import Path
import sys,json
ROOT=Path(sys.argv[1]);sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import jax
import numpy as np
from proofs.s1_km5.full_oracle import inputs
from gpuwrf.dynamics.les3d_smagorinsky import les3d_smagorinsky_memory
assert jax.devices()[0].platform=='cpu'
with np.load(ROOT/'data/fixtures/sms3dtke-full-wrf-v1.npz') as z:
    inp,cfg=inputs(z,'real_mixed')
    result=jax.eval_shape(lambda:les3d_smagorinsky_memory(inp,cfg))
summary=[dict(shape=list(x.shape),dtype=str(x.dtype)) for x in jax.tree.leaves(result)]
Path(sys.argv[2]).write_text(json.dumps(summary,indent=1));print('SHAPE_PASS',len(summary))
