"""Solar E78 through public native RestartStore and the real driver admission.

Authenticated damage is rehashed to exercise semantic admission rather than
the hash gate. Driver controls stop at admission; sentinel stepping and this
module are structural IO evidence, not a Solar physics/forecast proof.
"""
from dataclasses import replace
from datetime import datetime
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.diagnostics.precipitation import PrecipitationBuckets
from gpuwrf.diagnostics.solar_config import configuration
from gpuwrf.runtime.restart_store import RestartStore,nested_identity
from test_online_diagnostics import populated as populated_nwp
from test_online_diagnostics_restart_store import identity,rewrite
from test_solar_restart import case,_exact

GROUP="solar_interpolation"
CONFIG="solar_interpolation_config"
MARKER="solar_interpolation_schema_version"
PARAMETERS=("swddir_ref","bb","Bx","swdown_ref","gg","Gx","coszen_ref")
FLUXES=("swdown","swddir","swddni","swddif","gsw")
LEAVES=tuple(("parameters",name) for name in PARAMETERS)+tuple(("fluxes",name) for name in FLUXES)


@pytest.fixture
def native_case(case):
    grid,nl,carry=case
    nwp,radar=populated_nwp(carry.state)
    buckets=PrecipitationBuckets(*(np.full((4,4),i+.125,np.float32) for i in range(3)))
    nl=replace(nl,prec_acc_dt=15.,nwp_diagnostics=1,do_radar_ref=1,diagnostic_history_interval=30.)
    carry=carry.replace(precipitation_buckets=buckets,nwp_diagnostics=nwp,radar_reflectivity=radar)
    return grid,nl,carry


def exact_tree(left,right):
    assert jax.tree.structure(left)==jax.tree.structure(right)
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right),strict=True):
        a,b=np.asarray(a),np.asarray(b)
        assert (a.dtype,a.shape,a.tobytes())==(b.dtype,b.shape,b.tobytes())


def store_generation(path,nl,carry,**kwargs):
    store=RestartStore(path,10**7,2,0)
    generation,_=store.save({"d01":carry},{"d01":17},{"d01":nl.dt_s},identity(nl),**kwargs)
    return store,generation


def damage_leaf(payload,member,name,kind):
    group=payload["extra_fields"][GROUP]
    part=getattr(group,member)
    leaf=getattr(part,name)
    changed=(None if kind=="missing" else np.asarray(leaf,np.float64)
             if kind=="precision" else np.asarray(leaf)[:,:-1])
    payload["extra_fields"][GROUP]=group._replace(**{member:part._replace(**{name:changed})})


def damage_schema(payload,manifest,kind):
    configs=manifest["identity"]
    if kind=="marker":
        payload.pop(MARKER)
    elif kind=="version":
        payload[MARKER]=2
    elif kind=="config_missing":
        configs.pop(CONFIG)
    elif kind=="config_domains":
        configs[CONFIG]={}
    elif kind=="config_outer_type":
        # Same apparent domain set: must reject the non-Mapping before string
        # indexing, rather than leaking a TypeError from configs['d01'].
        configs[CONFIG]=["d01"]
    elif kind=="config_inner_type":
        configs[CONFIG]["d01"]=[]
    elif kind=="config_key":
        configs[CONFIG]["d01"].pop("radiation_interval_s")
    elif kind=="config_selector":
        configs[CONFIG]["d01"]["swint_opt"]=0
    elif kind=="config_clock":
        configs[CONFIG]["d01"]["radiation_cadence_steps"]=0
    elif kind=="group":
        payload["extra_fields"][GROUP]=None
    elif kind in ("parameters","fluxes"):
        group=payload["extra_fields"][GROUP]
        payload["extra_fields"][GROUP]=group._replace(**{kind:None})
    elif kind=="config_missing_deleted_group":
        configs.pop(CONFIG)
        payload["extra_fields"].pop(GROUP)
        payload["carry_field_order"].remove(GROUP)
    else:
        raise AssertionError(f"unknown test damage {kind}")


SCHEMA_DAMAGE=("marker","version","config_missing","config_domains","config_outer_type",
               "config_inner_type","config_key","config_selector","config_clock","group",
               "parameters","fluxes","config_missing_deleted_group")


@pytest.mark.parametrize("device",[False,True])
def test_public_native_solar_exact_positive_and_phase1_groups(native_case,tmp_path,device):
    _,nl,carry=native_case
    store,generation=store_generation(tmp_path,nl,carry)
    snapshot,manifest=store.read(generation,device=device,expected_identity=identity(nl))
    assert manifest["identity"][CONFIG]["d01"]==configuration(nl)
    restored=snapshot["carries"]["d01"]
    get_group=(lambda name:getattr(restored,name)) if device else (lambda name:restored["extra_fields"][name])
    _exact(carry.solar_interpolation,get_group(GROUP))
    for name in ("precipitation_buckets","nwp_diagnostics","radar_reflectivity"):
        exact_tree(getattr(carry,name),get_group(name))


@pytest.mark.parametrize("device",[False,True])
@pytest.mark.parametrize("kind",SCHEMA_DAMAGE)
def test_public_native_solar_schema_damage_refused(native_case,tmp_path,device,kind):
    _,nl,carry=native_case
    store,generation=store_generation(tmp_path,nl,carry)
    rewrite(generation,lambda payload,manifest:damage_schema(payload,manifest,kind))
    with pytest.raises(ValueError,match="Solar.*E78"):
        store.read(generation,device=device)


@pytest.mark.parametrize("device",[False,True])
@pytest.mark.parametrize("member,name",LEAVES)
@pytest.mark.parametrize("kind",["missing","precision","shape"])
def test_public_native_solar_each_leaf_damage_refused(native_case,tmp_path,device,member,name,kind):
    _,nl,carry=native_case
    store,generation=store_generation(tmp_path,nl,carry)
    rewrite(generation,lambda payload,manifest:damage_leaf(payload,member,name,kind))
    with pytest.raises(ValueError,match="Solar.*E78"):
        store.read(generation,device=device)


@pytest.mark.parametrize("member,name",LEAVES)
@pytest.mark.parametrize("kind",["precision","shape"])
def test_public_native_solar_save_refuses_each_bad_leaf(native_case,tmp_path,member,name,kind):
    from gpuwrf.io.restart import _carry_to_payload
    _,nl,carry=native_case
    payload=_carry_to_payload(carry)
    damage_leaf(payload,member,name,kind)
    broken=carry.replace(solar_interpolation=payload["extra_fields"][GROUP])
    with pytest.raises(ValueError,match="Solar.*E78"):
        store_generation(tmp_path,nl,broken)
    assert not list(tmp_path.glob("generation-*"))


@pytest.mark.parametrize("changes",[
    {"swint_opt":0},{"dt_s":20.},{"radiation_interval_s":660.},
    {"radiation_cadence_steps":34},{"ra_sw_physics":1},
])
def test_public_native_requested_solar_selector_and_both_clocks_match(native_case,tmp_path,changes):
    _,nl,carry=native_case
    store,generation=store_generation(tmp_path,nl,carry)
    with pytest.raises(ValueError,match="identity mismatch"):
        store.read(generation,expected_identity=identity(replace(nl,**changes)))


def test_nested_identity_binds_each_domains_solar_config(native_case,tmp_path):
    _,nl,_=native_case
    other=replace(nl,dt_s=54.,radiation_interval_s=1800.,radiation_cadence_steps=33)
    cfg=SimpleNamespace(input_dir=tmp_path,output_dir=tmp_path/"out",feedback=False,emit_initial_history=True)
    bundles={name:SimpleNamespace(namelist=selected) for name,selected in (("d01",nl),("d02",other))}
    actual=nested_identity(cfg,bundles,datetime(2023,1,15))
    assert actual[CONFIG]=={"d01":configuration(nl),"d02":configuration(other)}
    assert actual[CONFIG]["d01"]["radiation_interval_s"]==600.
    assert actual[CONFIG]["d02"]["radiation_interval_s"]==1800.


def test_inactive_solar_keeps_exact_phase1_groups_and_their_marker_guard(native_case,tmp_path):
    _,nl,carry=native_case
    nl=replace(nl,swint_opt=0)
    carry=carry.replace(solar_interpolation=None)
    store,generation=store_generation(tmp_path,nl,carry)
    restored=store.read(generation,expected_identity=identity(nl))[0]["carries"]["d01"]
    assert restored.solar_interpolation is None
    for name in ("precipitation_buckets","nwp_diagnostics","radar_reflectivity"):
        exact_tree(getattr(carry,name),getattr(restored,name))
    rewrite(generation,lambda payload,manifest:payload.pop("online_diagnostics_schema_version"))
    with pytest.raises(ValueError,match="online diagnostics schema.*E78"):
        store.read(generation)


def test_native_identity_requires_solar_map_even_with_all_solar_deleted(native_case,tmp_path):
    _,nl,carry=native_case
    nl=replace(nl,swint_opt=0)
    carry=carry.replace(solar_interpolation=None)
    store,generation=store_generation(tmp_path,nl,carry)
    rewrite(generation,lambda payload,manifest:damage_schema(payload,manifest,"config_missing_deleted_group"))
    with pytest.raises(ValueError,match="per-domain Solar configuration.*E78"):
        store.read(generation)


DRIVER_DAMAGE=("marker","version","group","precision","shape","config_missing_deleted_group")


@pytest.mark.parametrize("kind",["valid_active","valid_inactive",*DRIVER_DAMAGE])
def test_actual_native_driver_solar_admission_before_any_step(native_case,tmp_path,monkeypatch,kind):
    from tests.v025.restart.test_single_domain_restart import _install_driver,DT
    grid,nl,carry=native_case
    nl=replace(nl,dt_s=DT,radiation_cadence_steps=11)
    if kind=="valid_inactive":
        nl=replace(nl,swint_opt=0)
        carry=carry.replace(solar_interpolation=None)
    finals=[]
    pipeline,storage,_=_install_driver(monkeypatch,(grid,nl,carry),finals)
    bound={**identity(nl),"output_dir":str((tmp_path/"out").resolve())}
    admitted=[]
    def after_admission(carries,source):
        restored=carries["d01"]
        for name in (GROUP,"precipitation_buckets","nwp_diagnostics","radar_reflectivity"):
            exact_tree(getattr(carry,name),getattr(restored,name))
        admitted.append(source)
        raise RuntimeError("valid Solar carry admitted")
    monkeypatch.setattr(pipeline,"_require_native_base_state",after_admission)
    store=RestartStore(tmp_path/"store",10**7,2,0)
    generation,_=store.save({"d01":carry},{"d01":1},{"d01":DT},bound,driver_state={"forecast_hours":3.})
    valid=kind.startswith("valid_")
    if not valid:
        def corrupt(payload,manifest):
            if kind in ("precision","shape"):
                damage_leaf(payload,"parameters" if kind=="precision" else "fluxes",
                            "coszen_ref" if kind=="precision" else "swddni",kind)
            else:
                damage_schema(payload,manifest,kind)
        rewrite(generation,corrupt)
        # Authenticate the altered identity as well, so missing-config controls
        # reach the E78 guard instead of failing only the identity comparison.
        bound=json.loads((generation/"manifest.json").read_text())["identity"]
    monkeypatch.setattr(storage,"nested_identity",lambda *args:bound)
    config=pipeline.NestedPipelineConfig(tmp_path/"in",tmp_path/"out",tmp_path/"proof",3.,1,
        resume_checkpoint=generation,checkpoint_reserve_bytes=0,aot_prefetch=False)
    with pytest.raises(RuntimeError if valid else ValueError,
                       match="valid Solar carry admitted" if valid else "Solar.*E78"):
        pipeline.execute_nested_pipeline(config)
    assert len(admitted)==(1 if valid else 0)
    assert not finals,"Solar admission must finish before any step advances"
