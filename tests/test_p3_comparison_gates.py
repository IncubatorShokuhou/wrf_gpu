"""E200 fault injection through the proof CLIs; no column physics reruns."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROOF = Path(__file__).resolve().parents[1] / 'proofs/s2_p3'
sys.path.insert(0, str(PROOF))
import comparison
import literal_cpu
import run_column_gate

# Independent wrapper output inventory, including held NC for mp50.
COMMON = ('th', 'qv', 'qc', 'qr', 'nr', 'th_old', 'qv_old', 're_cloud',
          'ze', 'nc', 'qi', 'ni', 'qir', 'bir', 'vm', 'di', 'rhoi',
          're_ice', 'rainncv', 'snowncv', 'sr')
SECOND = ('qi2', 'ni2', 'qir2', 'bir2', 'vm2', 'di2', 'rhoi2')
COMPARED = (*COMMON, *SECOND, 'qzi')
NONFINITE = (np.nan, np.inf, -np.inf)


@pytest.fixture(scope='module')
def originals():
    result = {}
    for mp in (50, 51, 52, 53):
        with np.load(PROOF / 'fixtures' / f'columns_mp{mp}.npz', allow_pickle=False) as f:
            result[mp] = {k: f[k] for k in f.files}
    return result


@pytest.fixture
def run_gate(monkeypatch, tmp_path, originals):
    """Replay saved original arrays; inject faults before the real CLI checks."""
    for name in ('column', 'lookup', 'properties', 'process_support',
                 'initial_generated', 'processes_generated', 'final_generated'):
        module = importlib.import_module('gpuwrf.physics.p3.' + name)
        for attr in ('jnp', '_rounded_real', 'gamma', 'qzi_decode', 'qzi_encode'):
            if hasattr(module, attr):
                # literal_cpu.main installs a facade; restore it after each case.
                monkeypatch.setattr(module, attr, getattr(module, attr))
    column = importlib.import_module('gpuwrf.physics.p3.column')
    monkeypatch.setattr(column, 'selected_tables', lambda mp: {})
    monkeypatch.setattr(run_column_gate.jax, 'clear_caches', lambda: None)
    monkeypatch.setattr(np, 'savez_compressed', lambda *a, **kw: None)

    def run(kind, field='qir', target=None, bad=np.nan, mutate=None, spread=False, mp=None):
        mp = mp or (53 if field == 'qzi' else 52)
        truth = originals[mp]
        fixture = {k: v.copy() for k, v in truth.items()}
        literal = {k: v.copy() for k, v in truth.items()}
        fma = {k: v.copy() for k, v in truth.items()}
        names = (*COMMON, *SECOND) if mp == 52 else (*COMMON, 'qzi') if mp == 53 else COMMON
        output = {k: truth[k][..., 1].copy() for k in names}
        inventories = dict(fixture=fixture, literal=literal, fma=fma, candidate=output)
        if target:
            value = inventories[target][field]
            value.flat[1 if target != 'candidate' else 0] = np.float32(bad)
        if target == 'matching':  # handled below, before loading either array
            raise AssertionError('use mutate for matching operands')
        if mutate:
            mutate(inventories)

        class Archive(dict):
            @property
            def files(self):
                return list(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def load(path, **kwargs):
            name = Path(path).name
            source = literal if name.startswith('literal_') else fma if name.startswith('fma_') else fixture
            return Archive(source)

        calls = []

        def replay(fields, dt, it, *args, **kwargs):
            calls.append(it)
            # Each next call uses its original saved frame; fault is in call1.
            return output if it == 1 else {k: truth[k][..., it].copy() for k in names}

        monkeypatch.setattr(np, 'load', load)
        monkeypatch.setattr(column, '_step_impl', replay)
        monkeypatch.setattr(run_column_gate, 'step', replay)
        path = tmp_path / 'report.json'
        argv = ['--out', str(path), '--schemes', str(mp)]
        if kind == 'literal':
            try:
                literal_cpu.main(argv)
                accepted = True
            except AssertionError:
                accepted = False
        else:
            argv += ['--steps', '1']
            if spread or target in ('literal', 'fma'):
                argv += ['--wrf-spread', str(tmp_path)]
            accepted = run_column_gate.main(argv) == 0
        return accepted, json.loads(path.read_text()), calls

    return run


def rejected(result, *, nonfinite=True):
    accepted, report, _ = result
    assert not accepted
    assert report['verdict'] == 'FAIL'
    assert report['failure_count'] > 0
    if nonfinite:
        assert report['nonfinite_failures'] > 0
    if 'BYTE' in report:
        assert not report['BYTE']


@pytest.mark.parametrize('kind', ['literal', 'compiled'])
@pytest.mark.parametrize('field', COMPARED)
@pytest.mark.parametrize('bad', NONFINITE)
def test_every_candidate_output_rejects_one_nonfinite(run_gate, kind, field, bad):
    rejected(run_gate(kind, field, 'candidate', bad))


@pytest.mark.parametrize('kind', ['literal', 'compiled'])
@pytest.mark.parametrize('field', comparison.FIXTURE_FIELDS)
@pytest.mark.parametrize('bad', NONFINITE)
def test_entire_fixture_rejects_one_nonfinite_before_execution(run_gate, kind, field, bad):
    result = run_gate(kind, field, 'fixture', bad)
    rejected(result)
    assert result[2] == []


@pytest.mark.parametrize('field', COMPARED)
@pytest.mark.parametrize('bad', NONFINITE)
def test_literal_matching_nonfinite_never_counts_as_byte_pass(run_gate, field, bad):
    def matching(arrays):
        arrays['fixture'][field][..., 1].flat[0] = np.float32(bad)
        arrays['candidate'][field].flat[0] = np.float32(bad)
        assert arrays['fixture'][field][..., 1].tobytes() == arrays['candidate'][field].tobytes()
    rejected(run_gate('literal', field, mutate=matching))


@pytest.mark.parametrize('arm', ['literal', 'fma'])
@pytest.mark.parametrize('field', comparison.FIXTURE_FIELDS)
@pytest.mark.parametrize('bad', NONFINITE)
def test_every_auxiliary_field_rejects_one_nonfinite(run_gate, arm, field, bad):
    result = run_gate('compiled', field, arm, bad)
    rejected(result)
    assert result[2] == []


@pytest.mark.parametrize('target,kind', [('fixture', 'literal'), ('fixture', 'compiled'),
                                        ('candidate', 'literal'), ('candidate', 'compiled'),
                                        ('literal', 'compiled'), ('fma', 'compiled')])
@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype'])
def test_inventory_and_structure_are_required_by_both_clis(run_gate, target, kind, fault):
    def corrupt(arrays):
        data = arrays[target]
        if fault == 'missing':
            data.pop('qir')
        elif fault == 'extra':
            data['unexpected'] = data['qir']
        elif fault == 'shape':
            data['qir'] = data['qir'][:1]
        else:
            data['qir'] = data['qir'].astype(np.float64)
    rejected(run_gate(kind, mutate=corrupt, spread=True), nonfinite=False)


@pytest.mark.parametrize('bad', NONFINITE)
def test_corrupt_fma_cannot_hide_a_finite_rime_error(run_gate, bad):
    def corrupt(arrays):
        arrays['fma']['qir'][0, 0, 1] = np.float32(bad)
        arrays['candidate']['qir'].fill(0)
    rejected(run_gate('compiled', mutate=corrupt, spread=True))


@pytest.mark.parametrize('bad', NONFINITE)
def test_derived_limits_reject_nonfinite_before_error_comparison(run_gate, monkeypatch, bad):
    original = run_column_gate.compiled_limit

    def corrupt(name, *args):
        limit = original(name, *args)
        limit.flat[0] = bad
        return limit

    monkeypatch.setattr(run_column_gate, 'compiled_limit', corrupt)
    rejected(run_gate('compiled', spread=True))


def test_finite_operands_with_undefined_cloud_bound_fail(run_gate):
    def zero_mass(arrays):
        for arm in ('fixture', 'literal', 'fma'):
            arrays[arm]['dz'][0, 0, 0] = 0
    rejected(run_gate('compiled', mutate=zero_mass, spread=True))


@pytest.mark.parametrize('kind', ['literal', 'compiled'])
def test_finite_numerical_failure_is_counted(run_gate, kind):
    def omit_rime(arrays):
        arrays['candidate']['qir'].fill(0)
    accepted, report, _ = run_gate(kind, mutate=omit_rime, spread=True)
    assert not accepted
    assert report['failure_count'] > 0
    assert report['nonfinite_failures'] == 0
    assert report['frozen_failures'] > 0 if kind == 'literal' else report['schemes']['52']['steps']['1']['scores']['qir']['strict_bad'] > 0


@pytest.mark.parametrize('kind', ['literal', 'compiled'])
@pytest.mark.parametrize('mp', [50, 51, 52, 53])
def test_finite_saved_originals_pass_all_variant_inventories(run_gate, kind, mp):
    accepted, report, _ = run_gate(kind, spread=True, mp=mp)
    assert accepted
    assert report['verdict'] == 'PASS'
    assert report['failure_count'] == 0
