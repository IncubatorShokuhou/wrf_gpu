"""Registry-staggered force-down against independent source-literal WRF SINT."""
from dataclasses import dataclass,replace
from datetime import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.stochastic_config import bind_stochastic
from gpuwrf.nesting.boundary_construction import build_nest_force_weights
from gpuwrf.nesting.nest_o3 import force_child_carry_o3rad
from gpuwrf.physics.stochastic_driver import initial_stochastic,advance_stochastic


@dataclass(frozen=True)
class Carry:
    stochastic: object
    o3rad: object=None

    def replace(self,**kw):
        return replace(self,**kw)


@pytest.mark.parametrize("vertical",[0,1])
@pytest.mark.parametrize("batched",[False,True])
def test_source_sint_all_fields_child_holds_no_rng(vertical,batched):
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("wrf_sint",repo/"scripts/v0234_wrf_sint_source_oracle.py")
    oracle=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    control=dict(skebs=1,sppt=1,spp=1,rand_perturb=1,nens=7,
        skebs_vertstruc=vertical,sppt_vertstruc=vertical,rand_pert_vertstruc=vertical,
        vertstruc_spp_conv=vertical,vertstruc_spp_pbl=vertical,vertstruc_spp_lsm=vertical)
    control.update({"lengthscale_"+s:15000. for s in
        ("sppt","rand_pert","spp_conv","spp_pbl","spp_lsm")})
    parent_grid=SimpleNamespace(nx=17,ny=15,nz=4,dx=9000.,dy=9000.)
    child_grid=SimpleNamespace(nx=9,ny=9,nz=4,dx=3000.,dy=3000.)
    pc,cc=(bind_stochastic({"stoch":control},d) for d in ("d01","d02"))
    parent=initial_stochastic(pc,parent_grid,54,datetime(2026,10,13,18))
    child=initial_stochastic(cc,child_grid,18,datetime(2026,10,13,18))
    fixture=np.load(repo/"tests/fixtures/u2_stoch_real4.npz")
    patterns=[]
    for index,p in enumerate(parent.patterns):
        fields=[]
        for component in (("U","V") if index==0 else ("T",)):
            value=fixture[f"field_3_1_{component}"]
            if not vertical:
                value=value[:1]
            fields.append(jnp.asarray(value))
        patterns.append(p._replace(field=tuple(fields) if index==0 else fields[0]))
    parent=parent._replace(patterns=tuple(patterns))
    weights=build_nest_force_weights(parent_grid_ratio=3,i_parent_start=5,j_parent_start=4,
        parent_grid=parent_grid,child_grid=child_grid)
    one=lambda x:jax.tree.map(lambda a:jnp.stack((a,a)),x) if batched else x
    forced=force_child_carry_o3rad(Carry(one(child)),Carry(one(parent)),weights,
        parent_grid_ratio=3,batched=batched).stochastic
    if batched:
        forced=jax.tree.map(lambda x:x[0],forced)
    for index,(source,target) in enumerate(zip(parent.patterns,forced.patterns)):
        for component,(a,b) in enumerate(zip(source.field if index==0 else (source.field,),
                                             target.field if index==0 else (target.field,))):
            xs,ys=index==0 and component==0,index==0 and component==1
            ny,nx=9+int(ys),9+int(xs)
            truth=oracle.sint_full(np.asarray(a),ratio=3,i_parent_start=5,j_parent_start=4,
                child_ny=ny,child_nx=nx,xstag=xs,ystag=ys)
            np.testing.assert_allclose(np.asarray(b)[:,:ny,:nx],truth,rtol=4.e-6,
                atol=64*np.finfo(np.float32).eps*np.max(np.abs(truth)))
            np.testing.assert_array_equal(np.asarray(b)[:,ny:,:],0)
            np.testing.assert_array_equal(np.asarray(b)[:,:,nx:],0)
        assert target.sine.shape==target.cosine.shape==target.amplitude.shape==(0,0)
    before=forced
    after=jax.jit(lambda c:advance_stochastic(c,cc,dx=3000.,dy=3000.))(before)
    for a,b in zip(before.seeds,after.seeds):
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a,0)
    # Every non-SPPT held field remains bit-identical across child substeps.
    for i in (0,1,3,4,5,6):
        for a,b in zip(jax.tree.leaves(before.patterns[i]),jax.tree.leaves(after.patterns[i])):
            np.testing.assert_array_equal(a,b)
