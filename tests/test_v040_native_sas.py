"""Native SAS caller, WRF add_a2c momentum, cadence and restart contracts."""
import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import test_v040_sas94 as sas94
from gpuwrf.coupling import native_sas_adapter as adapter
from gpuwrf.runtime.operational_mode import _initial_carry_for_run, _physics_step_forcing


@pytest.mark.parametrize('scheme',(14,96))
def test_registry_namelist_binding_and_static_option_roundtrip(scheme):
    from gpuwrf.io.namelist_binding import bind_namelist_options,require_namelist_honoured
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    grid,state,nl=sas94._setup()
    raw={'physics':{'cu_physics':[scheme,scheme],'nsas_dx_factor':1,'sas_pgcon':[.3,.7]}}
    require_namelist_honoured(raw,('d01','d02'))
    for name,pgcon in (('d01',.3),('d02',.7)):
        bound=bind_namelist_options(nl,raw,name)
        leaves,tree=jax.tree.flatten(bound)
        restored=jax.tree.unflatten(tree,leaves)
        assert restored.cu_physics==scheme and restored.nsas_dx_factor==1 and restored.sas_pgcon==pgcon
    assert classify_scheme('cu_physics',scheme).status==SupportStatus.IMPLEMENTED


@pytest.mark.parametrize("scheme", (14, 96))
def test_time_n_caller_and_current_diagnostics(monkeypatch, scheme):
    grid, state, nl = sas94._setup()
    nz, ny, nx = state.theta.shape
    state = state.replace(pblh=jnp.full((ny,nx),321.,jnp.float32))
    diag = state.replace(pblh=jnp.full((ny,nx),654.,jnp.float32),
                         hfx=jnp.full((ny,nx),100.,jnp.float32),qfx=jnp.full((ny,nx),1.e-4,jnp.float32))
    captured = {}
    def kernel(*a, **kw):
        captured.update(a=a, kw=kw)
        return {**{k:jnp.zeros((ny*nx,nz),jnp.float32) for k in
                   ("RTHCUTEN","RQVCUTEN","RQCCUTEN","RQICUTEN","RUCUTEN","RVCUTEN")},
                "PRATEC":jnp.zeros(ny*nx,jnp.float32)}
    monkeypatch.setattr(adapter, f"cu_{'ksas14' if scheme==14 else 'nsas96'}_columns",kernel)
    rates = adapter.native_sas_rates(state,18.,grid,scheme=scheme,diagnostics=diag,
        stepcu=3,pgcon=.7,dx_factor_nsas=1,mp_physics=0)
    a, kw = captured["a"], captured["kw"]
    p = np.asarray(state.p,dtype=np.float32)
    pi = (p/np.float32(1.e5))**np.float32(2./7.)
    np.testing.assert_allclose(a[5],np.moveaxis(pi,0,-1).reshape(ny*nx,nz),rtol=2e-7)
    assert all(x.dtype == jnp.float32 for x in rates)
    assert not np.any(a[2]) and not np.any(a[3])
    assert kw["stepcu"] == 3 and kw["mp_physics"] == 0
    if scheme == 14:
        np.testing.assert_array_equal(a[12],np.full(ny*nx,654.))
        np.testing.assert_array_equal(a[13],np.full(ny*nx,321.))
        np.testing.assert_allclose(a[14],.5*(np.asarray(grid.eta_levels[:-1])+grid.eta_levels[1:]),rtol=1e-7)
        assert "pgcon" not in kw and "dx_factor_nsas" not in kw
    else:
        np.testing.assert_array_equal(a[13],np.full(ny*nx,654.))
        np.testing.assert_array_equal(a[14],np.full(ny*nx,100.))
        np.testing.assert_allclose(a[15],np.full(ny*nx,1.e-4),rtol=1e-7)
        assert kw["row_width"] == nx and kw["pgcon"] == .7 and kw["dx_factor_nsas"] == 1


@pytest.mark.parametrize("scheme", (14,96))
def test_cadence_slot_momentum_and_restart(monkeypatch, tmp_path, scheme):
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint_with_runtime_state
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
    grid, state, nl = sas94._setup()
    nl = dataclasses.replace(nl,cu_physics=scheme,nsas_dx_factor=1)
    carry = _initial_carry_for_run(state,nl)
    assert len(carry.cumulus_tendencies) == 9
    rates = tuple(jnp.full_like(x, .001 if i==6 else 1.e-6*(i+1),dtype=jnp.float32)
                  for i,x in enumerate(carry.cumulus_tendencies))
    captured = []
    def compute(entry,dt,grid,**kw):
        captured.append(kw)
        return rates
    monkeypatch.setattr(adapter,"native_sas_rates",compute)
    for step, refreshed in ((1,True),(2,False),(3,True),(4,False)):
        out, held = adapter.native_sas_cadence_step(state,state,carry.cumulus_tendencies,
            18.,grid,scheme=scheme,stepcu=3,itimestep=step,periodic_x=False,periodic_y=False)
        want = rates if refreshed else carry.cumulus_tendencies
        for a,b in zip(held,want): np.testing.assert_array_equal(a,b)
        # Centered interpolation check; specified/nested ranges are independently
        # gated against the original routines in test_v040_sas_boundaries.py.
        du, dv = 18.*np.asarray(want[7]),18.*np.asarray(want[8])
        eu, ev = np.array(state.u), np.array(state.v)
        eu[:,:,1:-1] += .5*(du[:,:,:-1]+du[:,:,1:])
        ev[:,1:-1,:] += .5*(dv[:,:-1,:]+dv[:,1:,:])
        np.testing.assert_array_equal(out.u,eu)
        np.testing.assert_array_equal(out.v,ev)
    forcing = _physics_step_forcing(carry,nl,0.,run_radiation=False)
    for a,b in zip(forcing.carry.cumulus_tendencies,rates): np.testing.assert_array_equal(a,b)
    assert captured[-1]["scheme"] == scheme
    assert captured[-1]["dx_factor_nsas"] == 1
    carry = forcing.carry
    p = tmp_path/"carry.pkl"
    write_checkpoint(carry.state,nl,grid,4,p,runtime_state=carry)
    _, restored_nl, _, step, restored = read_checkpoint_with_runtime_state(p)
    assert step == 4 and restored_nl.nsas_dx_factor == 1 and restored_nl.cu_physics == scheme
    for a,b in zip(restored.cumulus_tendencies,rates): np.testing.assert_array_equal(a,b)
    p = tmp_path/"wrfrst"
    write_wrfrst_carry(carry,grid,nl,p,valid_time=datetime(2023,1,15,0,1,12,tzinfo=timezone.utc),
                      run_start=datetime(2023,1,15,tzinfo=timezone.utc),lead_hours=.02,step_index=4)
    restored,_ = read_wrfrst_carry(p,namelist=nl)
    for a,b in zip(restored.cumulus_tendencies,rates):
        assert a.dtype == jnp.float32
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize("scheme", (14,94,96))
def test_restart_refuses_missing_or_truncated_held_rates(tmp_path, scheme):
    import pickle
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint_with_runtime_state
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
    grid,state,nl = sas94._setup()
    nl = dataclasses.replace(nl,cu_physics=scheme)
    carry = _initial_carry_for_run(state,nl)
    p = tmp_path/'carry.pkl'
    with pytest.raises(ValueError,match='E78'):
        write_checkpoint(state,nl,grid,4,p,runtime_state=carry.replace(cumulus_tendencies=None))
    write_checkpoint(state,nl,grid,4,p,runtime_state=carry)
    with p.open('rb') as f: payload=pickle.load(f)
    payload['runtime_state']=payload['runtime_state'].replace(cumulus_tendencies=None)
    with p.open('wb') as f: pickle.dump(payload,f)
    with pytest.raises(ValueError,match='E78'):
        read_checkpoint_with_runtime_state(p)
    p=tmp_path/'wrfrst'
    write_wrfrst_carry(carry,grid,nl,p,valid_time=datetime(2023,1,15,0,1,12,tzinfo=timezone.utc),
                      run_start=datetime(2023,1,15,tzinfo=timezone.utc),lead_hours=.02,step_index=4)
    # Corrupt the actual serialized pytree instead of testing a helper alone.
    with Dataset(p,'r+') as ds:
        import json
        manifest=json.loads(ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER)
        variable=manifest['cumulus_tendencies'][0]
        ds.variables['GPUWRF_CUMULUS_TENDENCIES_'+variable.upper()][:] = np.nan
    with pytest.raises(ValueError,match='E78'):
        read_wrfrst_carry(p,namelist=nl)


@pytest.mark.parametrize("scheme,active", ((14,31),(96,259)))
def test_complete_native_fixture_provenance_and_active_regimes(scheme, active):
    import hashlib
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/"proofs/v040/sas_family/fixtures"
    manifest=json.loads((root/f"native{scheme}_manifest.json").read_text())
    assert manifest["work"]["N"]==455 and manifest["work"]["KX"]==44
    assert manifest["qualification"]["literal_real8"]["all_pass"]
    # Retain raw native misses; admission follows the disclosed conditioning
    # policy and the final coupled D6, rather than silently loosening a gate.
    assert not manifest["qualification"]["native_real4"]["all_pass"]
    assert not manifest["qualification"]["tol_files_changed"]
    for name,sha in manifest["fixtures"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
        with np.load(root/name) as arrays:
            assert all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fiu")
            if "RAINCV" in arrays:
                assert arrays["RAINCV"].shape==(455,)
                assert np.count_nonzero(arrays["RAINCV"])==active
                assert np.count_nonzero(arrays["RAINCV"]==0.)==455-active
