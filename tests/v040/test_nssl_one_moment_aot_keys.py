"""The actual selected NSSL one-moment host constants must enter AOT identity."""
import json

import jax,jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.nssl2mom import constants
from gpuwrf.runtime import aot_cheap_key as ck,aot_executable as ax,aot_precompile as ap

jax.config.update('jax_enable_x64',True)


def namelist(mp):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    return OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=mp,
                               ra_sw_physics=0,ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)


def kernel(mode,field):
    @jax.jit
    def selected(x,nl):
        table=constants.get_one_moment_constants(mode,int(nl.mp_physics))
        value=table.ar if field=='ar' else table.cno[4]
        return x+value
    return selected


@pytest.mark.parametrize('mp',[19,21])
@pytest.mark.parametrize('mode',['fp32','fp64'])
@pytest.mark.parametrize('field',['ar','cno'])
def test_changed_resolved_one_moment_value_cannot_load_old_warm_blob(mp,mode,field,tmp_path,monkeypatch):
    R=np.float32 if mode=='fp32' else np.float64
    nl=namelist(mp);x=jnp.zeros(2,R);fn=kernel(mode,field)
    table=constants.get_one_moment_constants(mode,mp)
    original=getattr(table,field);value=original if field=='ar' else original[4]
    before=ck.cheap_key(fn,(x,nl),{},nl);assert before is not None
    lowered=fn.lower(x,nl)
    status=ap._serialize_domain_blob('nssl_one_moment',lowered.compile(),tmp_path,
                                    lowered=lowered,cheap_key=before,key_schema=ck.KEY_SCHEMA)
    assert status['aot_written'],status
    updated=R(np.nextafter(R(value),R(np.inf)))
    if field=='cno':
        changed=original.copy();changed[4]=updated
    else:changed=updated
    monkeypatch.setattr(table,field,changed)
    fn.clear_cache()  # The JIT captures the host provider during tracing.
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert after is not None and after!=before,'changed actual one-moment table value collided in cheap/AOT key'
    loaded,status=ap.load_domain_blob('nssl_one_moment',tmp_path,cheap_key=after,return_status=True)
    assert loaded is None,'new table values must not load the old constant executable'
    fresh=fn.lower(x,nl)
    assert ax.hlo_sha256_from_lowered(fresh)!=ax.hlo_sha256_from_lowered(lowered)
    np.testing.assert_array_equal(fn(x,nl),np.full(2,updated,R))
    fn.clear_cache()


@pytest.mark.parametrize('mp',[19,21])
def test_actual_asset_reload_changes_only_selected_variant_key(mp,tmp_path,monkeypatch):
    asset=tmp_path/'init_constants.json';asset.write_bytes(constants._ASSET.read_bytes())
    one=tmp_path/'one_moment_constants.json'
    data=json.loads(constants._ASSET.with_name(one.name).read_text());one.write_text(json.dumps(data))
    monkeypatch.setattr(constants,'_ASSET',asset)
    def clear():
        constants.get_constants.cache_clear();constants.get_one_moment_constants.cache_clear()
    clear()
    try:
        x=jnp.zeros(2,jnp.float32);fn=kernel('fp32','ar');nl=namelist(mp)
        before=ck.cheap_key(fn,(x,nl),{},nl)
        def plain(x,nl):return x
        inactive=ck.cheap_key(plain,(x,namelist(18)),{},namelist(18))
        other=21 if mp==19 else 19
        unrelated=ck.cheap_key(plain,(x,namelist(other)),{},namelist(other))
        original=np.float32(data[str(mp)]['fp32']['ar'][0])
        data[str(mp)]['fp32']['ar'][0]=float(np.nextafter(original,np.float32(np.inf)))
        one.write_text(json.dumps(data));clear();fn.clear_cache()
        after=ck.cheap_key(fn,(x,nl),{},nl)
        assert before is not None and after is not None and before!=after
        assert inactive==ck.cheap_key(plain,(x,namelist(18)),{},namelist(18)), 'inactive one-moment assets must not invalidate18'
        assert unrelated==ck.cheap_key(plain,(x,namelist(other)),{},namelist(other))
    finally:
        clear()


@pytest.mark.parametrize('mp',[19,21])
@pytest.mark.parametrize('name',['_CWR','_GRAD'])
def test_active_one_moment_efficiency_globals_replacement_changes_key(mp,name,monkeypatch):
    from gpuwrf.physics.nssl2mom import gs_a
    nl=namelist(mp);x=jnp.zeros(2,jnp.float32)
    @jax.jit
    def fn(x,nl):return x+jnp.asarray(getattr(gs_a,name),jnp.float32)[0,0]
    before=ck.cheap_key(fn,(x,nl),{},nl);assert before is not None
    hlo=ax.hlo_sha256_from_lowered(fn.lower(x,nl))
    table=np.asarray(getattr(gs_a,name),np.float32).copy()
    table[0,0]=np.nextafter(table[0,0],np.float32(np.inf))
    monkeypatch.setattr(gs_a,name,tuple(tuple(map(float,row)) for row in table))
    fn.clear_cache()
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert after is not None and before!=after,'actual active efficiency table replacement collided'
    assert hlo!=ax.hlo_sha256_from_lowered(fn.lower(x,nl))
    np.testing.assert_array_equal(fn(x,nl),np.full(2,table[0,0],np.float32))
    fn.clear_cache()
