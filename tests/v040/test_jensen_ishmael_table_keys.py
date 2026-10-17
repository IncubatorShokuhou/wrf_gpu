"""Changed actual cached host tables and argument maps cannot reuse warm AOT."""
from pathlib import Path
import json
import jax,jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics.jensen_ishmael import tables,level_support
from gpuwrf.runtime import aot_cheap_key as ck,aot_executable as ax,aot_precompile as ap

jax.config.update('jax_enable_x64',True)


def namelist():
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    return OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=55,
                               ra_sw_physics=0,ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('field',['itab','itabr','gamma_tab','igrdata','coltabn'])
def test_actual_cached_table_value_changes_key_HLO_and_warm_blob(dtype,field,tmp_path,monkeypatch):
    raw=tables._host_tables();raw_name=field if field!='coltabn' else 'coltabn_'+('fp32' if dtype==np.float32 else 'fp64')
    array=raw[raw_name];index=int(np.argmax(array));assert array.flat[index]>0
    @jax.jit
    def fn(x,nl):return x+tables.get_tables(dtype)[field].flat[index]
    nl=namelist();x=jnp.zeros(2,dtype=dtype)
    before=ck.cheap_key(fn,(x,nl),{},nl);assert before is not None
    old=np.asarray(fn(x,nl));lowered=fn.lower(x,nl)
    status=ap._serialize_domain_blob('jensen_table',lowered.compile(),tmp_path,
                                     lowered=lowered,cheap_key=before,key_schema=ck.KEY_SCHEMA)
    assert status['aot_written'],status
    # Change at the coarser raw/selected precision so promotion or demotion
    # cannot erase the perturbation (IGR is stored at original REAL8 precision).
    R=array.dtype.type if array.dtype.itemsize<=np.dtype(dtype).itemsize else dtype
    changed=array.copy();changed.flat[index]=np.nextafter(R(array.flat[index]),R(np.inf));changed.flags.writeable=False
    monkeypatch.setitem(raw,raw_name,changed);fn.clear_cache()
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert after is not None and after!=before,'Actual cached table values collided in cheap/AOT key'
    loaded,status=ap.load_domain_blob('jensen_table',tmp_path,cheap_key=after,return_status=True)
    assert loaded is None
    fresh=fn.lower(x,nl);assert ax.hlo_sha256_from_lowered(fresh)!=ax.hlo_sha256_from_lowered(lowered)
    assert np.all(np.asarray(fn(x,nl))!=old)
    fn.clear_cache()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_actual_bound_original_argument_map_changes_key_and_program(dtype,monkeypatch):
    mode='fp32' if dtype==np.float32 else 'fp64'
    folder=Path(__file__).resolve().parents[2]/'proofs/s2_mpmore/jensen55/aggregation'
    m=json.loads((folder/'manifest.json').read_text())
    with np.load(folder/(mode+'.npz')) as data:inputs=data['inputs'].copy()
    fields={n:inputs[:,i] for i,n in enumerate(m['input_names'])}
    data=tables.get_tables(dtype)
    args=tuple(fields[n] for n in level_support._AGG_NAMES)+(data['coltab'],data['coltabn'],*(np.zeros(len(inputs),dtype) for _ in range(6)))
    @jax.jit
    def fn(x,nl):return x+level_support._call('aggregation',args,{},jnp)[6][0]
    nl=namelist();x=jnp.zeros(2,dtype=dtype)
    before=ck.cheap_key(fn,(x,nl),{},nl);assert before is not None
    original=np.asarray(fn(x,nl));hlo=ax.hlo_sha256_from_lowered(fn.lower(x,nl))
    mapping=list(level_support._AGG_NAMES);mapping[1],mapping[12]=mapping[12],mapping[1]
    monkeypatch.setattr(level_support,'_AGG_NAMES',mapping);fn.clear_cache()
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert after is not None and before!=after,'Actual bound original-caller mapping collided in key'
    assert hlo!=ax.hlo_sha256_from_lowered(fn.lower(x,nl))
    assert np.any(np.asarray(fn(x,nl))!=original)
    fn.clear_cache()


def test_actual_registry_input_map_changes_key_and_program(monkeypatch):
    from gpuwrf.physics.jensen_ishmael import registry
    @jax.jit
    def fn(x,nl):
        # The initialized QICE3/QICE2 source selected by the actual loader map.
        return x+{'QICE3':np.float32(.125),'QICE2':np.float32(.25)}[registry.input_variables()['qi3']]
    nl=namelist();x=jnp.zeros(2,jnp.float32)
    before=ck.cheap_key(fn,(x,nl),{},nl);old=np.asarray(fn(x,nl))
    hlo=ax.hlo_sha256_from_lowered(fn.lower(x,nl))
    monkeypatch.setitem(registry.INPUT_VARIABLES,'qi3','QICE2');fn.clear_cache()
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert before is not None and before!=after,'Actual Jensen input map collided in key'
    assert hlo!=ax.hlo_sha256_from_lowered(fn.lower(x,nl))
    assert np.all(np.asarray(fn(x,nl))!=old)
    fn.clear_cache()
