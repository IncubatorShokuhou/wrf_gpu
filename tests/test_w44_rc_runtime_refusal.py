"""RC host admission: an unqualified family must never be partially applied."""
import json
from types import SimpleNamespace

import pytest

from gpuwrf import runtime_parameters as params


def test_loader_refuses_diffusion_with_qualification_reason(tmp_path):
    path = tmp_path / 'params.json'
    path.write_text(json.dumps({'schema': params.SCHEMA, 'domains': {
        'd01': {'diffusion_parameters': {'dampcoef': .2}}}}))
    with pytest.raises(ValueError, match=r'not available in v0\.4\.4: failed the GPU qualification; see KNOWN_ISSUES'):
        params.load_parameters(path)


def test_mapping_refuses_before_any_family_is_applied(monkeypatch):
    calls = []
    def bind(namelist, values):
        calls.append(values)
        return namelist
    monkeypatch.setattr(params, 'bind_mynn_parameters', bind)
    monkeypatch.setattr(params, 'bind_diffusion_parameters', bind)
    domains = {'d01': {'mynn_parameters': {'sqfac': 3.},
                       'diffusion_parameters': {'dampcoef': .2}}}
    with pytest.raises(ValueError, match='failed the GPU qualification'):
        params.bind_domain_parameters(SimpleNamespace(), domains, 'd01')
    assert calls == []


def test_one_policy_constant_can_refuse_mynn_too(monkeypatch, tmp_path):
    monkeypatch.setattr(params, 'UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES',
                        frozenset({'diffusion_parameters', 'mynn_parameters'}))
    path = tmp_path / 'params.json'
    path.write_text(json.dumps({'schema': params.SCHEMA, 'domains': {
        'd01': {'mynn_parameters': {'sqfac': 3.}}}}))
    with pytest.raises(ValueError, match='mynn_parameters not available'):
        params.load_parameters(path)


def test_cli_help_withdraws_diffusion_and_marks_mynn_pending():
    from gpuwrf.cli import build_parser
    parser = build_parser()
    subparsers = next(a for a in parser._actions if a.dest == 'command')
    action = next(a for a in subparsers.choices['run']._actions if a.dest == 'runtime_parameters')
    assert 'MYNN' in action.help and 'pending' in action.help
    assert 'diffusion' not in action.help and 'damping' not in action.help


@pytest.mark.parametrize("family", ["diffusion_parameters", "mynn_parameters"])
def test_direct_binders_consult_release_policy_before_device_allocation(monkeypatch, family):
    monkeypatch.setattr(params, 'UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES', frozenset({family}))
    binder = params.bind_diffusion_parameters if family == 'diffusion_parameters' else params.bind_mynn_parameters
    with pytest.raises(ValueError, match=family + ' not available'):
        binder(None, {'dampcoef': .4} if family == 'diffusion_parameters' else {'sqfac': 3.})


@pytest.mark.parametrize("family", ["diffusion_parameters", "mynn_parameters"])
def test_direct_namelist_consults_same_release_policy(monkeypatch, family):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    monkeypatch.setattr(params, 'UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES', frozenset({family}))
    # Invalid rad_diag proves the release guard precedes other namelist consumers.
    with pytest.raises(ValueError, match=family + ' not available'):
        OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., rad_diag=9,
                            **{family: {'dampcoef': .4} if family == 'diffusion_parameters' else {'sqfac': 3.}})


def test_direct_namelist_unset_default_remains_static():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., dampcoef=.2)
    assert nl.diffusion_parameters is None and nl.mynn_parameters is None and nl.dampcoef == .2
