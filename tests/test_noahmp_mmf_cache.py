"""MMF cache regression only; pristine-WRF fidelity is in the oracle tests."""
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.runtime.aot_cheap_key import cheap_key,static_config_hash
from gpuwrf.runtime.aot_executable import hlo_sha256_from_lowered
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.contracts.grid import GridSpec


def test_mmf_static_selector_uses_merged_key_mechanism():
    nl=OperationalNamelist(GridSpec.canary_3km_template(),None,None,15.,
        use_noahmp=True,noahmp_mmf_enabled=True,mp_physics=0,ra_sw_physics=0,
        ra_lw_physics=0,sf_sfclay_physics=0,bl_pbl_physics=0,cu_physics=0)
    @jax.jit
    def kernel(x,nml):
        return x*(1 if nml.noahmp_mmf_opt_inf==1 else 2) if nml.noahmp_mmf_enabled else x+3
    x=jnp.arange(4,dtype=jnp.float32)
    variants=[nl,replace(nl,noahmp_mmf_opt_inf=2),replace(nl,noahmp_mmf_enabled=False)]
    keys=[cheap_key(kernel,(x,n),{},n) for n in variants]
    assert all(k is not None for k in keys)
    assert len({static_config_hash(n) for n in variants})==3
    assert all(keys[i]!=keys[j] for i in range(3) for j in range(i))
    assert len({hlo_sha256_from_lowered(kernel.lower(x,n)) for n in variants})==3
    # No new baked table provider: all MMF geography/FSATMX are operands.
    assert len({baked_table_hash(kernel,n) for n in variants})==1


def test_fsatmx_is_a_runtime_operand_and_same_key_reuse_is_correct(tmp_path):
    from gpuwrf.io.noahmp_mmf import _fsatmx
    from gpuwrf.physics.noahmp.groundwater import _soilwater5_core
    from test_noahmp_mmf_oracle import column_args
    from gpuwrf.integration.init_kernels import _StaticKwargs
    fixture=Path(__file__).resolve().parents[1]/'proofs/s2_mmf/original_wrf_real4.npz'
    with np.load(fixture) as data:
        args=jax.tree.map(jnp.asarray,column_args(data,1,'rain'))
    table=tmp_path/'MPTABLE.TBL'
    values=[]
    for value in (.38,.76):
        table.write_text(f'&noahmp_global_parameters\n FSATMX={value},\n/\n')
        values.append(jnp.float32(_fsatmx(table)))
    original_args=list(args);original_args[16]=values[0];args=tuple(original_args)
    changed=list(args);changed[16]=values[1];changed=tuple(changed)
    kernel=jax.jit(partial(_soilwater5_core,opt_inf=1))
    empty=_StaticKwargs(())
    first=cheap_key(kernel,args,{},empty);second=cheap_key(kernel,changed,{},empty)
    assert first is not None and first==second
    a,b=kernel.lower(*args),kernel.lower(*changed)
    assert hlo_sha256_from_lowered(a)==hlo_sha256_from_lowered(b)
    executable=a.compile()
    original=executable(*args);reused=executable(*changed);fresh=b.compile()(*changed)
    assert np.any(np.asarray(original[6])!=np.asarray(reused[6]))  # RUNSF is active.
    for left,right in zip(jax.tree.leaves(reused),jax.tree.leaves(fresh),strict=True):
        left,right=np.asarray(left),np.asarray(right)
        assert np.isfinite(left).all() and left.tobytes()==right.tobytes()
