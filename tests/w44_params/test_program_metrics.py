import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / 'proofs/w44_params/program_metrics.py'
SPEC = importlib.util.spec_from_file_location('params_program_metrics', PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def test_hardlink_alias_binds_actual_program_and_missing_proof_refuses(tmp_path):
    run, entries = {}, []
    for label, count in (('off', 8), ('on', 9)):
        key = label + '_key'
        run[label + '_aot'] = {'domains': {'d01': {'cheap_key': key}}}
        path = '/cache/aot/tag/d01/' + label + '_hlo.xlaexec'
        entries.append({'path': path, 'program': 'd01', 'blob_sha256': label,
                        'aliases': ['/cache/aot/tag/d01/k_' + key + '.xlaexec'],
                        'kinput_transpose_status': 'MEASURED_RICH_AOT_HLO'})
        token = hashlib.sha256(path.encode()).hexdigest()[:12]
        (tmp_path / ('hlo_scan_' + token + '.json')).write_text(json.dumps({
            'n_fusions': count, 'transpose_hero_fusions': [
                {'kind': 'kInput', 'n_instr': 40, 'n_transpose': 2}]}))
    gate = {'e41_gate': 'PASS', 'entries': entries}
    assert METRICS.compare(run, gate, tmp_path)['on_minus_off']['d01']['n_fusions'] == 1
    entries[1]['aliases'] = []
    with pytest.raises(AssertionError, match='E41 artifact'):
        METRICS.compare(run, gate, tmp_path)


def test_incomplete_e41_never_becomes_a_structural_pass(tmp_path):
    with pytest.raises(AssertionError, match='complete'):
        METRICS.compare({}, {'e41_gate': 'INCOMPLETE'}, tmp_path)
