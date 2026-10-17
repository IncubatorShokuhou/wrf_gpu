"""Distinct SAS94 ARW port vs pristine WRF, plus caller/cadence/restart gates."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import test_v013_operational_smoke as smoke
from gpuwrf.coupling import sas94_adapter as adapter
from gpuwrf.physics.cumulus_sas94 import cu_sas94_columns
from gpuwrf.runtime.operational_mode import (
    _initial_carry_for_run, _physics_step_forcing, _resolve_operational_suite,
)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "proofs/v040/sas_family/fixtures"
INPUT = ROOT / "proofs/v034/scalesas/savepoints/columns_inputs.npz"


def _column_args(path=INPUT):
    inp = np.load(path)
    return [inp[k] for k in ("T","QV","QC","QI","P","PI","RHO","DZ","U","V","W")] + [inp["P8W"][:,0], inp["XLAND"]]


def test_fixtures_tables_and_independent_active_regimes():
    man = json.loads((FIX/"manifest.json").read_text())
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == man["input_sha256"]
    for name, sha in man["fixture_sha256"].items():
        assert hashlib.sha256((FIX/name).read_bytes()).hexdigest() == sha
    assert hashlib.sha256((FIX/man["rare_input"]).read_bytes()).hexdigest() == man["rare_input_sha256"]
    for name, sha in man["rare_fixture_sha256"].items():
        assert hashlib.sha256((FIX/name).read_bytes()).hexdigest() == sha
    for cfg in man["configs"].values():
        assert cfg["r4"]["active_deep"] >= 200
        assert cfg["r4"]["inactive_deep"] >= 200
        assert cfg["r4"]["shallow_changed_columns"] >= 100
        assert cfg["r4"]["ice_source_columns"] > 100
    assert man["tolerances"]["exclusions"] == []
    tables = json.loads((FIX.parent/"gfs_sas_tables_manifest.json").read_text())
    for mode, build in tables["builds"].items():
        path = ROOT/f"src/gpuwrf/physics/gfs_sas_tables_{mode}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == build["sha256"]
        with np.load(path) as arrays:
            assert all(np.isfinite(a).all() and a.dtype == np.float64 for a in arrays.values())


def test_table_cache_is_host_only_across_independent_jits():
    from gpuwrf.physics.gfs_sas_shallow import _funcs, _tables
    for mode in ("r4","r8"):
        a = jax.jit(lambda p:_funcs(mode)[0](p))(jnp.asarray(1000.,jnp.float64))
        b = jax.jit(lambda p:_funcs(mode)[0](p+1.))(jnp.asarray(1000.,jnp.float64))
        assert a.dtype == b.dtype == jnp.float64
        assert np.isfinite(a) and np.isfinite(b) and b > a
        assert all(isinstance(x,np.ndarray) and not x.flags.writeable for x in _tables(mode).values())


@pytest.mark.parametrize("cfg,dt,stepcu", (("dt54_stepcu5",54.,5),("dt18_stepcu1",18.,1)))
@pytest.mark.parametrize("mode", ("r4","r8"))
def test_pristine_cu_sas_default_deep_and_shallow(cfg,dt,stepcu,mode):
    args = [np.concatenate((a,b),axis=0) for a,b in zip(_column_args(),_column_args(FIX/"rare_inputs.npz"))]
    actual = jax.jit(functools.partial(cu_sas94_columns, dt=dt,stepcu=stepcu,literals=mode))(*args)
    with np.load(FIX/f"{cfg}_{mode}.npz") as main_expected, np.load(FIX/f"rare_{cfg}_{mode}.npz") as rare_expected:
        expected = {k:np.concatenate((main_expected[k],rare_expected[k]),axis=0) for k in main_expected}
        for key, value in actual.items():
            a, b = np.asarray(value), expected[key]
            assert np.isfinite(a).all() and np.isfinite(b).all(), key
            if key in ("HBOT","HTOP"):
                np.testing.assert_array_equal(a,b)
            elif mode == "r4":
                assert a.dtype == np.float32
                ulp = np.spacing(np.maximum(np.abs(a),np.abs(b)))
                assert np.all(np.abs(a.astype(np.float64)-b) <= ulp), key
                assert np.mean(a == b) >= .999, key
            else:
                scale = (np.max(np.abs(b),axis=1,keepdims=True) if a.ndim == 2 else np.abs(b))+1e-300
                assert np.all(np.abs(a-b) <= 1e-9*scale), key


@pytest.mark.parametrize("cfg,dt,stepcu", (("dt54_stepcu5",54.,5),("dt18_stepcu1",18.,1)))
@pytest.mark.parametrize("mode", ("r4","r8"))
def test_pristine_work_rejection_witnesses(cfg,dt,stepcu,mode):
    """Ten updraft/jmin-work and two downdraft-work rejections from pristine census."""
    actual = jax.jit(functools.partial(cu_sas94_columns,dt=dt,stepcu=stepcu,literals=mode))(
        *_column_args(FIX/"rare_inputs.npz"))
    with np.load(FIX/f"rare_{cfg}_{mode}.npz") as expected:
        assert len(expected["RAINCV"]) == 12 and np.count_nonzero(expected["RAINCV"]) == 0
        for key,value in actual.items():
            a,b = np.asarray(value),expected[key]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            if key in ("HBOT","HTOP"):
                np.testing.assert_array_equal(a,b)
            elif mode == "r4":
                assert np.all(np.abs(a.astype(np.float64)-b) <= np.spacing(np.maximum(np.abs(a),np.abs(b)))),key
                # BYTE fraction is gated over the complete467 corpus above.
            else:
                scale = (np.max(np.abs(b),axis=1,keepdims=True) if a.ndim == 2 else np.abs(b))+1e-300
                assert np.all(np.abs(a-b) <= 1e-9*scale),key


def _setup():
    grid = smoke._grid(nz=44,ny=2,nx=2)
    state = smoke._convective_state(grid)
    nml = smoke._namelist(grid,dt_s=18.,mp_physics=0,bl_pbl_physics=0,
                          sf_sfclay_physics=0,cu_physics=94,ra_sw_physics=0,ra_lw_physics=0)
    return grid,state,dataclasses.replace(nml,cumulus_cadence_steps=3,sas_pgcon=.7)


def _active_setup(scheme=94):
    """Oracle-active column with its coherent dry hydrostatic mass coordinate.

    The generic smoke uses a flat MU unrelated to its p/ph fields. SAS reads
    WRF phy_prep hydrostatic pressure, so that sounding is not a SAS rain test.
    This maps a pristine-oracle-active sounding to State and matches dry mass
    thickness to its P8W/Q profiles without patching any production operator.
    """
    grid,state,nml = _setup()
    reference = "native14_r4.npz" if scheme in (14,99) else "dt18_stepcu1_r4.npz"
    with np.load(INPUT) as inp, np.load(FIX/reference) as ref:
        i = int(np.argmax(ref["PRATEC"]))
        assert ref["PRATEC"][i] > .001
        nz,ny,nx = state.theta.shape
        col = lambda name:jnp.asarray(np.broadcast_to(inp[name][i][:,None,None],(nz,ny,nx)),jnp.float32)
        qv = col("QV")
        # This physics-slot smoke disables microphysics (MP0), whose Registry
        # contains passive QV only. Keep that absent-species state faithful.
        qc,qi = jnp.zeros_like(qv),jnp.zeros_like(qv)
        theta = col("T")/col("PI")*(1.+jnp.asarray(461.6/287.,jnp.float32)*qv)
        qt = inp["QV"][i].astype(np.float64)
        dry_dp = -np.diff(inp["P8W"][i].astype(np.float64))/(1.+qt)
        mu = dry_dp.sum()
        dnw = -dry_dp/mu
        z = np.concatenate([[0.],np.cumsum(inp["DZ"][i].astype(np.float64))])
        ph = jnp.asarray(np.broadcast_to((9.81*z)[:,None,None],(nz+1,ny,nx)),jnp.float32)
        state = state.replace(theta=theta,qv=qv,qc=qc,qi=qi,qr=jnp.zeros_like(qv),qs=jnp.zeros_like(qv),
            qg=jnp.zeros_like(qv),p_total=col("P"),ph_total=ph,
            u=jnp.asarray(np.broadcast_to(inp["U"][i][:,None,None],(nz,ny,nx+1)),jnp.float32),
            v=jnp.asarray(np.broadcast_to(inp["V"][i][:,None,None],(nz,ny+1,nx)),jnp.float32),
            w=jnp.asarray(np.broadcast_to(inp["W"][i][:,None,None],(nz+1,ny,nx)),jnp.float32),
            xland=jnp.full((ny,nx),inp["XLAND"][i],jnp.float32),
            mu_total=jnp.full((ny,nx),mu,jnp.float32),
            pblh=jnp.full((ny,nx),750.,jnp.float32))
        metrics = dataclasses.replace(grid.metrics,c1h=jnp.ones(nz,jnp.float32),c2h=jnp.zeros(nz,jnp.float32),
            dnw=jnp.asarray(dnw,jnp.float32),p_top=jnp.asarray(inp["P8W"][i,-1],jnp.float32))
    # These native source witnesses use DX=9000m. The generic 3km smoke
    # changes KSAS's scale-aware trigger, so retain the oracle's actual work.
    projection = dataclasses.replace(grid.projection,dx_m=9000.,dy_m=9000.) if scheme in (14,99) else grid.projection
    grid = dataclasses.replace(grid,metrics=metrics,projection=projection)
    nml = dataclasses.replace(nml,grid=grid,metrics=metrics,cu_physics=scheme,cumulus_cadence_steps=1,sas_pgcon=.55)
    return grid,state,nml


def test_opt_in_static_pgcon_and_both_pipeline_binding():
    from gpuwrf.io.namelist_binding import bind_namelist_options, require_namelist_honoured
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    grid,state,nml = _setup()
    _resolve_operational_suite(nml)
    spec = {"physics":{"cu_physics":[94,94],"sas_pgcon":[.3,.7]}}
    require_namelist_honoured(spec,("d01","d02"))
    for name,want in (("d01",.3),("d02",.7)):
        bound = bind_namelist_options(nml,spec,name)
        assert bound.cu_physics == 94 and bound.sas_pgcon == want
        leaves,td = jax.tree.flatten(bound)
        assert jax.tree.unflatten(td,leaves).sas_pgcon == want
    default = dataclasses.replace(nml,sas_pgcon=.55)
    assert len(nml.tree_flatten()[1]) == len(default.tree_flatten()[1])+1
    assert classify_scheme("cu_physics",94).status is SupportStatus.IMPLEMENTED
    assert "GPU/coupled-forecast qualification pending" in classify_scheme("cu_physics",94).reason
    # Both production loaders use the same binding authority. Their actual load
    # is also exercised by the lane's real Swiss binding/coupled probe.
    for path in ("nested_pipeline.py","daily_pipeline.py"):
        source = (ROOT/"src/gpuwrf/integration"/path).read_text()
        assert "bind_namelist_options(" in source


@pytest.mark.parametrize("pgcon",(.55,.7))
@pytest.mark.parametrize("stochastic",(False,True))
@pytest.mark.parametrize("tracers",(False,True))
@pytest.mark.parametrize("ocean",(False,True))
def test_pgcon_aux_composes_with_observation_lake_and_ccn_tails(pgcon,stochastic,tracers,ocean):
    _,_,base = _setup()
    marker = object()
    nml = dataclasses.replace(base,sas_pgcon=pgcon,observation_nudging=marker,
                              lakedepth_default=20.,ccn_conc=5e7)
    from gpuwrf.io.stochastic_config import bind_stochastic
    from gpuwrf.tracers.config import TracerSpecies,UserTracerConfig
    stoch=bind_stochastic({'stoch':{'sppt':1}},'d01') if stochastic else None
    tracer=UserTracerConfig((TracerSpecies('DYE'),),(),'a'*64) if tracers else None
    nml=dataclasses.replace(nml,stochastic_config=stoch,tracer_config=tracer,
        windfarm_opt=1,windfarm=marker,nsas_dx_factor=1,osas_domain_id=7)
    if ocean:
        from gpuwrf.physics.ocean_mixed_layer import initialize
        initial=initialize(jnp.full((nml.grid.ny,nml.grid.nx),290.,jnp.float32),50.)
        nml=dataclasses.replace(nml,sf_ocean_physics=1,ocean_initial=initial)
    leaves,tree = jax.tree.flatten(nml)
    restored = jax.tree.unflatten(tree,leaves)
    assert restored.sas_pgcon == pgcon and restored.observation_nudging is marker
    assert restored.stochastic_config is stoch and restored.tracer_config is tracer
    assert restored.windfarm is marker and restored.nsas_dx_factor==1 and restored.osas_domain_id==7
    assert restored.sf_ocean_physics==int(ocean)
    if ocean:
        for a,b in zip(jax.tree.leaves(restored.ocean_initial),jax.tree.leaves(initial),strict=True):
            np.testing.assert_array_equal(a,b)
    assert restored.lakedepth_default == 20. and restored.ccn_conc == 5e7


def test_adapter_reads_dynamic_exner_and_hydro_pressure(monkeypatch):
    # Recorded CPU-WRF phy_prep values distinguish dynamic and hydrostatic P.
    row = json.loads((ROOT/"tests/v025/b_phys/fixtures/mynn_phy_exner_h1.json").read_text())
    raw = {k:np.asarray(row[k],np.float32) for k in ("p_dyn","p_hyd","pi_phy","theta","qv","qc","qi")}
    b,k = raw["p_dyn"].shape
    field = lambda a:jnp.asarray(a.T[:,:,None],jnp.float32)
    qv = raw["qv"]
    theta = raw["theta"]*(np.float32(1.)+np.float32(461.6/287.)*qv)
    zface = np.broadcast_to(np.arange(k+1,dtype=np.float32)[:,None,None]*100.,(k+1,b,1))
    state = SimpleNamespace(p=field(raw["p_dyn"]),theta=field(theta),qv=field(qv),
        qc=field(raw["qc"]),qi=field(raw["qi"]),ph=jnp.asarray(zface*np.float32(9.81)),
        u=jnp.ones((k,b,2),jnp.float32)*3.,v=jnp.ones((k,b+1,1),jnp.float32)*4.,
        w=jnp.zeros((k+1,b,1),jnp.float32),xland=jnp.ones((b,1),jnp.float32))
    hp = field(raw["p_hyd"])
    psfc = jnp.full((b,1),100000.,jnp.float32)
    rho = jnp.full((k,b,1),.9,jnp.float32)
    monkeypatch.setattr(adapter,"_wrf_hydrostatic_pressure_profiles_from_state",lambda *a,**kw:(hp,jnp.concatenate([psfc[None],hp]),psfc))
    monkeypatch.setattr(adapter,"_wrf_phy_prep_rho_from_state",lambda *a,**kw:rho)
    captured = {}
    def kernel(*a,**kw):
        captured.update(args=a,kwargs=kw)
        return {**{name:jnp.zeros((b,k),jnp.float32) for name in ("RTHCUTEN","RQVCUTEN","RQCCUTEN","RQICUTEN")},"PRATEC":jnp.zeros(b,jnp.float32)}
    monkeypatch.setattr(adapter,"cu_sas94_columns",kernel)
    adapter.sas94_rates(state,18.,SimpleNamespace(metrics=object()),stepcu=3,pgcon=.7)
    a = captured["args"]
    # pi_phy is an independent pristine CPU fixture, not the adapter under test.
    np.testing.assert_allclose(a[5],raw["pi_phy"],rtol=2e-7,atol=0.)
    np.testing.assert_array_equal(a[4],raw["p_hyd"])
    dry = theta/(np.float32(1.)+np.float32(461.6/287.)*qv)
    np.testing.assert_allclose(a[0],dry*raw["pi_phy"],rtol=3e-7,atol=0.)
    np.testing.assert_array_equal(a[6],np.full((b,k),.9,np.float32))
    np.testing.assert_array_equal(a[11],np.full(b,100000.,np.float32))
    assert captured["kwargs"]["pgcon"] == .7 and a[-1] == 3


@pytest.mark.parametrize("mp,qc,qi",((0,False,False),(1,True,False),(3,True,False),(8,True,True)))
def test_adapter_honours_registry_species_presence(monkeypatch,mp,qc,qi):
    grid,state,nml = _setup()
    state = state.replace(qc=jnp.full_like(state.qc,.002),qi=jnp.full_like(state.qi,.001))
    captured={}
    def kernel(*a,**kw):
        captured.update(qc=np.asarray(a[2]),qi=np.asarray(a[3]))
        shape=a[0].shape
        return {**{name:jnp.ones(shape,jnp.float32)*1e-8 for name in
                   ("RTHCUTEN","RQVCUTEN","RQCCUTEN","RQICUTEN")},"PRATEC":jnp.zeros(shape[0],jnp.float32)}
    monkeypatch.setattr(adapter,"cu_sas94_columns",kernel)
    rates=adapter.sas94_rates(state,18.,grid,mp_physics=mp)
    assert bool(np.any(captured["qc"])) == qc and bool(np.any(captured["qi"])) == qi
    assert bool(np.any(np.asarray(rates[2]))) == qc and bool(np.any(np.asarray(rates[3]))) == qi


def test_cadence_and_operational_slot_use_time_n_and_held_rates(monkeypatch):
    grid,state,nml = _setup()
    state = jax.tree.map(lambda x:x.astype(jnp.float32) if hasattr(x,"dtype") and x.dtype.kind=="f" else x,state)
    rates = tuple(jnp.full(state.theta.shape,x,jnp.float32) for x in (1e-8,2e-8,0.,0.)) + (jnp.full(state.theta.shape[1:],.001,jnp.float32),)
    monkeypatch.setattr(adapter,"sas94_rates",lambda entry,dt,grid,**kw:rates)
    carry = _initial_carry_for_run(state,nml)
    assert all(np.asarray(x).dtype == np.float32 for x in carry.cumulus_tendencies)
    old = tuple(jnp.zeros_like(x) for x in carry.cumulus_tendencies)
    for step,refresh in ((1,True),(2,False),(3,True),(4,False),(6,True)):
        out,held = adapter.sas94_cadence_step(state,state,old,18.,grid,stepcu=3,itimestep=step,pgcon=.7)
        want = adapter._held_layout(rates) if refresh else old
        for a,b in zip(held,want):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(np.asarray(out.rainc_acc)-np.asarray(state.rainc_acc),18.*np.asarray(want[6]),rtol=2e-7,atol=1e-8)
    f = _physics_step_forcing(carry,nml,0.,run_radiation=False)
    for a,b in zip(f.carry.cumulus_tendencies,adapter._held_layout(rates)):np.testing.assert_array_equal(a,b)
    assert np.max(np.asarray(f.state.rainc_acc)) > 0.
    assert np.array_equal(np.asarray(f.state.u),np.asarray(state.u))


def test_existing_carry_and_static_option_roundtrip_both_restart_formats(tmp_path):
    from datetime import datetime,timezone
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    grid,state,nml = _setup()
    carry = _initial_carry_for_run(state,nml)
    held = tuple(jnp.full_like(x,.001*(i+1)) for i,x in enumerate(carry.cumulus_tendencies))
    carry = carry.replace(cumulus_tendencies=held)
    p = tmp_path/"carry.pkl"
    write_checkpoint(state,nml,grid,4,p,runtime_state=carry)
    _,restored_nml,_,step,restored = read_checkpoint_with_runtime_state(p)
    assert step == 4 and restored_nml.sas_pgcon == .7 and restored_nml.cu_physics == 94
    for a,b in zip(restored.cumulus_tendencies,held):np.testing.assert_array_equal(a,b)
    p = tmp_path/"wrfrst"
    write_wrfrst_carry(carry,grid,nml,p,valid_time=datetime(2023,1,15,0,1,12,tzinfo=timezone.utc),
                      run_start=datetime(2023,1,15,tzinfo=timezone.utc),lead_hours=.02,step_index=4)
    restored,_ = read_wrfrst_carry(p,namelist=nml)
    for a,b in zip(restored.cumulus_tendencies,held):
        assert np.asarray(a).dtype == np.float32
        np.testing.assert_array_equal(a,b)
