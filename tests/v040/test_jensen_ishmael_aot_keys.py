"""Selected Jensen's actual bound IGR tuple participates in AOT identity."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics.jensen_ishmael import math
from gpuwrf.runtime import aot_cheap_key as ck,aot_executable as ax,aot_precompile as ap

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_actual_bound_growth_table_cannot_load_old_warm_blob(dtype,tmp_path,monkeypatch):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    nl=OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=55,
                          ra_sw_physics=0,ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)
    @jax.jit
    def fn(t,nl):return math.get_igr(t)
    x=jnp.asarray([271.65],dtype=dtype)
    before=ck.cheap_key(fn,(x,nl),{},nl);assert before is not None
    lowered=fn.lower(x,nl);original=np.asarray(fn(x,nl))
    status=ap._serialize_domain_blob('jensen_igr',lowered.compile(),tmp_path,
                                     lowered=lowered,cheap_key=before,key_schema=ck.KEY_SCHEMA)
    assert status['aot_written'],status
    bound=math.get_igr.__kwdefaults__['data']
    monkeypatch.setitem(math.get_igr.__kwdefaults__,'data',(bound[0]*.9,*bound[1:]))
    fn.clear_cache()
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert after is not None and after!=before,'Actual bound IGR values collided in cheap/AOT key'
    loaded,status=ap.load_domain_blob('jensen_igr',tmp_path,cheap_key=after,return_status=True)
    assert loaded is None
    fresh=fn.lower(x,nl)
    assert ax.hlo_sha256_from_lowered(fresh)!=ax.hlo_sha256_from_lowered(lowered)
    assert np.all(np.asarray(fn(x,nl))!=original)
    fn.clear_cache()
