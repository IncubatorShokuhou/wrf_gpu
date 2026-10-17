"""Different actual baked NPZ/CO2/calendar values must not share executables."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.aot_cheap_key import cheap_key
from gpuwrf.physics import ra_flg,ra_gfdleta,gfdleta_column
from gpuwrf.coupling.flg_gfdl_radiation import GfdlGrid


def config(code,grid_key=None):
    return OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,
        ra_lw_physics=code,ra_sw_physics=code,cu_physics=0,sf_surface_physics=4,
        time_utc=datetime(2026,1,15,12,tzinfo=timezone.utc),gfdl_grid=grid_key)


@pytest.fixture
def provider(monkeypatch):
    root=Path(__file__).resolve().parents[3]
    with np.load(root/'tests/fixtures/s2_gfdl_cloud_v1.npz',allow_pickle=False) as f:
        full=f['in_znw'].copy();half=f['in_znu'][:-1].copy();top=np.asarray(f['in_scalars'][0],np.float32)
    with np.load(root/'tests/fixtures/s2_gfdl_radtn_v1.npz',allow_pickle=False) as f:
        co2={k[4:]:f[k].copy() for k in f.files if k.startswith('co2_')}
        domain={k[7:]:f[k].copy() for k in f.files if k.startswith('domain_')}
    monkeypatch.setattr(gfdleta_column,'co2_tables',lambda *_a,**_k:(co2,domain))
    return GfdlGrid(full.tobytes(),half.tobytes(),top.tobytes()),co2,domain


@pytest.mark.parametrize('code,port,name',[(7,ra_flg,'ozone'),(99,ra_gfdleta,'ferrier_tbpvs')])
def test_host_npz_values_change_real_cheap_key_and_compiled_constant(monkeypatch,provider,code,port,name):
    nl=config(code,provider[0] if code==99 else None)
    original=port._host_tables();keys=[];outputs=[];x=jnp.asarray(0.,jnp.float32)
    @jax.jit
    def kernel(value,namelist):
        return value+jnp.asarray(port._host_tables()[name].reshape(-1)[0],jnp.float32)
    for offset in (0.,.25):
        replacement=dict(original);table=original[name].copy();table.flat[0]+=np.float32(offset);replacement[name]=table
        monkeypatch.setattr(port,'_host_tables',lambda:replacement)
        kernel.clear_cache()
        keys.append(cheap_key(kernel,(x,nl),{},nl));outputs.append(float(kernel(x,nl)))
    assert all(keys) and keys[0]!=keys[1]
    assert outputs[0]!=outputs[1]


def test_actual_bound_co2_changes_provider_factory_and_key(monkeypatch,provider):
    grid,co2,domain=provider;nl=config(99,grid);x=jnp.asarray(0.,jnp.float32)
    @jax.jit
    def kernel(value,namelist):
        return value+jnp.float32(namelist.gfdl_grid.baked_tables()['spectral']['co251'][0,0])
    records=[]
    for offset in (0.,.125):
        replacement=dict(co2);table=co2['co251'].copy();table[0,0]+=np.float32(offset);replacement['co251']=table
        monkeypatch.setattr(gfdleta_column,'co2_tables',lambda *_a,**_k:(replacement,domain))
        kernel.clear_cache()
        records.append((cheap_key(kernel,(x,nl),{},nl),float(kernel(x,nl)),id(grid.solver())))
    assert records[0][0]!=records[1][0] and all(r[0] for r in records)
    assert records[0][1]!=records[1][1]
    assert records[0][2]!=records[1][2],'cached factory retained old native provider'


def test_initial_calendar_separates_jax_and_aot_keys_without_manual_clear():
    nl=config(7);x=jnp.asarray(0.,jnp.float32)
    @jax.jit
    def kernel(value,namelist):return value+jnp.float32(namelist.time_utc.year)
    variants=[replace(nl,time_utc=datetime(year,1,15,12,tzinfo=timezone.utc)) for year in (2025,2026)]
    keys=[cheap_key(kernel,(x,n),{},n) for n in variants]
    assert all(keys) and len(set(keys))==2
    assert [float(kernel(x,n)) for n in variants]==[2025.,2026.]
