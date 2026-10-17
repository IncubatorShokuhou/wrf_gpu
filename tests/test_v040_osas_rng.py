"""Fixed OSAS counter RNG against independent integer reference and JAX primitive."""
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.osas_rng import initial_osas_key,osas_draws,threefry2x32

jax.config.update('jax_enable_x64',True)


def reference(key,counter):
    # Plain Python integer implementation; no JAX helper or product imports.
    mask=(1<<32)-1
    keys=[int(key[0]),int(key[1]),int(key[0])^int(key[1])^0x1BD11BDA]
    a,b=(int(counter[0])+keys[0])&mask,(int(counter[1])+keys[1])&mask
    rotate=[13,15,26,6,17,29,16,24]
    for group in range(5):
        for r in rotate[(group%2)*4:(group%2+1)*4]:
            a=(a+b)&mask
            b=(((b<<r)&mask)|(b>>(32-r)))^a
        a=(a+keys[(group+1)%3])&mask
        b=(b+keys[(group+2)%3]+group+1)&mask
    return a,b


def test_threefry_known_answer_and_independent_integer_reference():
    # Random123 Threefry2x32-20 zero key / zero counter known-answer vector.
    a,b=threefry2x32(jnp.zeros(2,jnp.uint32),0,0)
    assert (int(a),int(b))==(0x6b200159,0x99ba4efe)
    key=initial_osas_key(3)
    counters=np.array([[0,0],[1,0],[1,234],[2**32-1,2**32-1],[54321,777777]],np.uint32)
    expected=np.array([reference(np.asarray(key),pair) for pair in counters],np.uint32)
    actual=jax.jit(threefry2x32)(key,counters[:,0],counters[:,1])
    np.testing.assert_array_equal(np.stack(actual,axis=1),expected)
    # Independent library implementation of the published permutation.
    from jax._src.prng import threefry2x32_p
    other=threefry2x32_p.bind(key[0],key[1],jnp.asarray(counters[:,0]),jnp.asarray(counters[:,1]))
    np.testing.assert_array_equal(np.stack(other,axis=1),expected)


def test_draw_key_counter_partition_and_restart_replay():
    import pickle
    key=initial_osas_key(2)
    columns=np.arange(455,dtype=np.uint32)
    draw=jax.jit(osas_draws)(key,1234,columns)
    bits=np.array([reference(np.asarray(key),(1234,c)) for c in columns],np.uint32)
    expected=bits[:,0].astype(np.float64)*2.**-32+(bits[:,1]>>11).astype(np.float64)*2.**-53
    np.testing.assert_array_equal(draw,expected)
    assert np.isfinite(draw).all() and np.all((draw>=0.)&(draw<1.))
    restored=pickle.loads(pickle.dumps(np.asarray(key)))
    np.testing.assert_array_equal(draw,osas_draws(restored,1234,columns))
    np.testing.assert_array_equal(draw[100:200],osas_draws(key,1234,columns[100:200]))
    for different in (osas_draws(initial_osas_key(3),1234,columns),
                      osas_draws(key,1235,columns),osas_draws(key,1234,columns+1)):
        assert np.all(np.asarray(draw)!=np.asarray(different))


def test_operational_domain_binding_keys_and_all_restart_formats(tmp_path):
    import dataclasses
    from datetime import datetime,timezone
    from types import SimpleNamespace
    import pytest
    import test_v040_sas94 as sas94
    from gpuwrf.io.namelist_binding import bind_namelist_options
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.runtime.restart_store import RestartStore,nested_identity
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    grid,state,nl=sas94._setup()
    ns=bind_namelist_options(nl,{'physics':{'cu_physics':[95,95]}},'d02')
    assert ns.osas_domain_id==2
    leaves,tree=jax.tree.flatten(ns)
    assert jax.tree.unflatten(tree,leaves).osas_domain_id==2
    carry=_initial_carry_for_run(state,ns)
    np.testing.assert_array_equal(carry.osas_rng_key,np.array([950040,2],np.uint32))
    expected=osas_draws(carry.osas_rng_key,1234,np.arange(4,dtype=np.uint32))
    p=tmp_path/'checkpoint'
    write_checkpoint(state,ns,grid,1234,p,runtime_state=carry)
    restored=read_checkpoint_with_runtime_state(p)[4]
    np.testing.assert_array_equal(osas_draws(restored.osas_rng_key,1234,np.arange(4,dtype=np.uint32)),expected)
    write_restart(carry,ns,grid,1234,p)
    np.testing.assert_array_equal(read_restart(p)[0].osas_rng_key,carry.osas_rng_key)
    p=tmp_path/'wrfrst'
    write_wrfrst_carry(carry,grid,ns,p,valid_time=datetime(2023,1,15,6,10,12,tzinfo=timezone.utc),
                      run_start=datetime(2023,1,15,tzinfo=timezone.utc),lead_hours=6.17,step_index=1234)
    # Stored domain selection alone must authenticate keys when no namelist is supplied.
    np.testing.assert_array_equal(read_wrfrst_carry(p)[0].osas_rng_key,carry.osas_rng_key)
    config=NestedPipelineConfig(tmp_path,tmp_path/'out',tmp_path/'proof',hours=1,max_dom=1)
    ident=nested_identity(config,{'d02':SimpleNamespace(namelist=ns)},datetime(2023,1,15,tzinfo=timezone.utc))
    store=RestartStore(tmp_path/'store',10**7,2,0)
    generation,_=store.save({'d02':carry},{'d02':1234},{'d02':18.},ident)
    restored=store.read(generation,expected_identity=ident)[0]['carries']['d02']
    np.testing.assert_array_equal(restored.osas_rng_key,carry.osas_rng_key)
    for bad in (None,jnp.array([950040,3],jnp.uint32),jnp.array([950040,2],jnp.int32)):
        invalid=carry.replace(osas_rng_key=bad)
        with pytest.raises(ValueError,match='E78'):
            write_checkpoint(state,ns,grid,1234,p,runtime_state=invalid)
        with pytest.raises(ValueError,match='E78'):
            store.save({'d02':invalid},{'d02':1234},{'d02':18.},ident)
