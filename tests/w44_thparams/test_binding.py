"""Shared family keys authenticate names, trace values and select the real adapter."""
from dataclasses import replace
import json
import jax
import numpy as np
import pytest
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf import runtime_parameters as registry
from gpuwrf.physics.thompson_parameters import PARAMETERS, EXPERIMENTAL_PARAMETERS
from gpuwrf.coupling.thompson_runtime_adapter import selected_adapter, thompson_runtime_adapter
from gpuwrf.coupling.physics_couplers import thompson_adapter


@pytest.fixture
def original(monkeypatch):
    for suffix in ("NATIVE_REAL","FULL_COLUMN","MIXED_PHASE_WRF","COLUMN_SED","SED_FP32","WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_"+suffix,"1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED","0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING","0")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER","1")
    return OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,run_physics=True,mp_physics=8)


def bind(namelist,values):
    return registry.bind_domain_parameters(namelist,{"d01":{"thompson_parameters":values}},"d01")


def test_values_share_keys_names_do_not(original):
    from gpuwrf.runtime.aot_cheap_key import static_config_hash,carry_aval_hash
    defaults={name:row[0] for name,row in PARAMETERS.items() if name not in EXPERIMENTAL_PARAMETERS}
    a,b=bind(original,defaults),bind(original,{k:2*v for k,v in defaults.items()})
    assert jax.tree.structure(a)==jax.tree.structure(b)!=jax.tree.structure(original)
    assert static_config_hash(a)==static_config_hash(b)!=static_config_hash(original)
    assert carry_aval_hash((a,),{})==carry_aval_hash((b,),{})
    assert static_config_hash(bind(original,{"tno":5}))!=static_config_hash(bind(original,{"ato":.304}))
    assert len(jax.tree.leaves(a))==len(jax.tree.leaves(original))+len(defaults)
    back=jax.tree.unflatten(jax.tree.structure(a),jax.tree.leaves(a))
    assert registry.restart_parameter_digests(back)==registry.restart_parameter_digests(a)
    assert registry.restart_parameter_digests(a)!=registry.restart_parameter_digests(b)
    assert registry.restart_parameter_digests(original)=={}
    assert original.tree_flatten()[1]==replace(original,thompson_parameters=None).tree_flatten()[1]


def test_original_function_object_off_and_operands_on(original):
    assert selected_adapter(original) is thompson_adapter
    bound=bind(original,{"tno":10})
    selected=selected_adapter(bound)
    assert selected.func is thompson_runtime_adapter
    assert selected.keywords["parameters"] is bound.thompson_parameters


def test_off_keeps_the_operational_adapter_seam(original):
    def operational_spy(*args, **kwargs):
        pytest.fail("the selector must return the caller's adapter, not call it")
    assert selected_adapter(original, operational_spy) is operational_spy


@pytest.mark.parametrize("name", sorted(EXPERIMENTAL_PARAMETERS))
def test_unqualified_controls_are_refused_by_the_public_binder(original, name):
    with pytest.raises(ValueError, match="experimental and refused"):
        bind(original, {name: PARAMETERS[name][0]})
    with pytest.raises(ValueError, match="experimental and refused"):
        selected_adapter(replace(original, thompson_parameters={name: PARAMETERS[name][0]}))


def test_json_registry_entry(original,tmp_path):
    path=tmp_path/"parameters.json"
    data={"d01":{"thompson_parameters":{"av_c":63389200,"tno":10,"ato":.608}}}
    path.write_text(json.dumps({"schema":registry.SCHEMA,"domains":data}))
    assert registry.load_parameters(path)==data
    assert registry.bind_domain_parameters(original,path,"d02") is original
    bound=registry.bind_domain_parameters(original,path,"d01")
    assert float(bound.thompson_parameters["av_c"])==63389200


@pytest.mark.parametrize("changes",[{"mp_physics":28},{"mp_physics":16},{"run_physics":False}])
def test_inactive_or_wrong_scheme_refused(original,changes):
    with pytest.raises(ValueError,match="active mp_physics=8"):
        bind(replace(original,**changes),{"tno":5})


def test_legacy_driver_refused(original,monkeypatch):
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER","0")
    with pytest.raises(ValueError,match="WRF-ordered"):
        bind(original,{"tno":5})


def test_catalog_and_restart_helpers(original):
    from gpuwrf.io.scheme_catalog import RUNTIME_PARAMETER_FAMILIES
    from gpuwrf.runtime.restart_parameters import static_namelist,parameter_digests,validate_parameters
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    assert set(RUNTIME_PARAMETER_FAMILIES["thompson_parameters"]["parameters"])==set(PARAMETERS)
    a,b=bind(original,{"av_c":31694600}),bind(original,{"av_c":63389200})
    assert static_namelist(original) is original
    assert canonical_digest(static_namelist(a))==canonical_digest(static_namelist(b))
    assert parameter_digests(original)=={}
    assert parameter_digests(a)!=parameter_digests(b)
    validate_parameters(parameter_digests(a),a)
    with pytest.raises(ValueError,match="differ from checkpoint"):
        validate_parameters(parameter_digests(a),b)
