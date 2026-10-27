"""Release admission for W2 failures; these tests never use research fixtures."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.surface_params import controls as sc
from gpuwrf.surface_params import cli as spcli

W2 = ('bexp', 'dksat', 'hvb', 'hvt', 'psisat', 'rsmin', 'smcmax', 'snow_emis', 'z0sno')
REASON = r'not available in v0\.4\.4: failed the GPU qualification; see KNOWN_ISSUES'


class UnreadableValue:
    def __array__(self, *args, **kwargs):
        raise AssertionError('a value was consumed before release admission')


class NoBinding:
    def __iter__(self):
        raise AssertionError('binding started before every member was admitted')


@pytest.mark.parametrize('name', W2)
def test_each_w2_control_is_refused_by_real_release_policy(name):
    # Deleting the guard gives nine genuine DID NOT RAISE assertion failures.
    assert set(sc.UNAVAILABLE_SURFACE_CONTROLS) == set(W2)
    with pytest.raises(ValueError, match=REASON):
        sc.refuse_unqualified_controls({name: UnreadableValue()})


@pytest.mark.parametrize('name', W2)
@pytest.mark.parametrize('boundary', ['expand', 'values', 'prior', 'validate', 'bind', 'host_query', 'aux', 'apply', 'lsm_apply', 'clock', 'nl_clock', 'namelist'])
def test_all_direct_boundaries_reject_before_values_or_other_validation(name, boundary):
    bad = {name: UnreadableValue()}
    source = {'d01': {'z0mvt': UnreadableValue()}, 'd02': bad}
    with pytest.raises(ValueError, match=REASON):
        if boundary == 'expand': sc.expand_controls(None, bad)
        elif boundary == 'values': sc.expand_controls(None, values=bad)
        elif boundary == 'prior': sc.prior_controls(None, [name])
        elif boundary == 'validate': sc.validate_controls(bad, None)
        elif boundary == 'bind': sc.bind_domain_controls(None, source, 'd01')
        elif boundary == 'host_query': sc.has_host_controls(source)
        elif boundary == 'aux': sc.controls_aux(bad)
        elif boundary == 'apply': sc.apply_surface_controls(None, None, None, bad)
        elif boundary == 'lsm_apply':
            from gpuwrf.surface_params.lsmparams import apply
            apply(None, None, bad)
        elif boundary == 'clock':
            sc.traced_controls(SimpleNamespace(surface_controls=None), SimpleNamespace(surface_controls=bad))
        elif boundary == 'nl_clock':
            sc.traced_controls(SimpleNamespace(surface_controls=bad), SimpleNamespace(surface_controls=None))
        else:
            from gpuwrf.runtime.operational_mode import OperationalNamelist
            # rad_diag invalid: release refusal must run before all other checks.
            OperationalNamelist(None, None, None, 18., rad_diag=9, surface_controls=bad)


@pytest.mark.parametrize('name', W2)
def test_writer_rejects_whole_request_before_any_array_or_file(name, tmp_path):
    out = tmp_path / 'controls.npz'
    with pytest.raises(ValueError, match=REASON):
        sc.save_controls(out, {'d01': {'z0mvt': UnreadableValue()}, 'd02': {name: UnreadableValue()}})
    assert not out.exists() and not out.with_name(out.name + '.tmp').exists()


@pytest.mark.parametrize('name', W2)
def test_loader_refuses_authenticated_old_control_files(name, tmp_path):
    fields = {name: np.ones((1, 1), np.float32)}
    # Construct an authenticated retained/research file without the public writer.
    manifest = {'schema': sc.SCHEMA, 'domains': {'d02': {
        'names': [name], 'digest': sc.controls_digest(fields),
        'shapes': {name: [1, 1]}, 'dtypes': {name: 'float32'}}}}
    path = tmp_path / 'old.npz'
    np.savez(path, manifest=np.frombuffer(json.dumps(manifest).encode(), np.uint8), **{'d02/'+name: fields[name]})
    with pytest.raises(ValueError, match=REASON): sc.load_controls(path)
    with pytest.raises(ValueError, match=REASON): sc.bind_domain_controls(None, path, 'd01')
    from gpuwrf.integration.nested_pipeline import _bind_batch_surface_controls
    with pytest.raises(ValueError, match=REASON):
        _bind_batch_surface_controls(NoBinding(), ('d01',),
            [{'d01': {'z0mvt': UnreadableValue()}}, path], 2)


@pytest.mark.parametrize('name', W2)
def test_batch_refuses_late_member_before_any_lane_binding(name):
    from gpuwrf.integration.nested_pipeline import _bind_batch_surface_controls
    with pytest.raises(ValueError, match=REASON):
        _bind_batch_surface_controls(NoBinding(), ('d01',),
            [{'d01': {'z0mvt': UnreadableValue()}}, {'d01': {name: UnreadableValue()}}], 2)


@pytest.mark.parametrize('name', W2)
def test_basis_and_eki_requests_refuse_before_partial_work(name):
    with pytest.raises(ValueError, match=REASON):
        spcli.expand_from_basis(None, {'domains': {}}, {'d01': {'z0mvt': 1.}, 'd02': {name: 1.}})
    spec = {'parameters': [{'name': name, 'control': name}]}
    with pytest.raises(ValueError, match=REASON): spcli.member_multipliers(spec, UnreadableValue())
    with pytest.raises(ValueError, match=REASON): spcli._write_members(spec, [], None, None, 0)


def test_cli_expand_and_eki_reject_with_no_partial_outputs(tmp_path, capsys):
    basis = tmp_path / 'basis.npz'
    np.savez(basis, manifest=np.frombuffer(json.dumps({'schema': spcli.BASIS_SCHEMA, 'domains': {}}).encode(), np.uint8))
    output = tmp_path / 'out.npz'
    assert spcli.main_args(['expand', '--basis', str(basis), '--set', 'rsmin=1.1', '--output', str(output)]) == 2
    assert 'failed the GPU qualification' in capsys.readouterr().err and not output.exists()
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'parameters': [{'name': 'hvt', 'control': 'hvt'}]}))
    directory = tmp_path / 'ensemble'
    assert spcli.main_args(['eki', 'init', '--spec', str(spec), '--basis', str(basis), '--members', '2',
                           '--seed', '1', '--output-dir', str(directory)]) == 2
    assert 'failed the GPU qualification' in capsys.readouterr().err and not directory.exists()


def test_basis_discloses_unavailable_controls_and_legacy_stays_usable(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'proofs/w44_lsmparams'))
    from columns import build
    nl = build()[5]
    prior = sc.prior_controls(nl, device=False)
    assert set(prior) == set(sc.CONTROLS) and not set(prior) & set(W2)
    manifest = spcli.write_basis({'d01': nl}, tmp_path / 'basis.npz')
    for name in W2:
        entry = manifest['domains']['d01']['controls'][name]
        assert entry['active'] is False and 'failed the GPU qualification' in entry['reason']
    assert manifest['domains']['d01']['controls']['z0mvt']['active']
    path = tmp_path / 'legacy.npz'
    sc.save_controls(path, {'d01': prior})
    assert sc.load_controls(path)[0]['d01'].keys() == prior.keys()


def test_legacy_and_host_names_are_not_refused_and_default_none_stays_none():
    sc.refuse_unqualified_controls(set(sc.CONTROLS) | set(sc.HOST_CONTROLS))
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.)
    assert nl.surface_controls is None
    assert sc.traced_controls(nl, SimpleNamespace(surface_controls=None)) is None


def test_help_preserves_trackp_and_discloses_w2_refusal():
    from gpuwrf.cli import build_parser
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == 'command')
    action = next(a for a in sub.choices['run']._actions if a.dest == 'surface_controls')
    assert 'z0mvt' in action.help and 'W2' in action.help and 'unavailable in v0.4.4' in action.help
