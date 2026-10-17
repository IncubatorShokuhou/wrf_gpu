"""Trace original fresh10/11 ice-number stages; cached original bodies only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from run import CACHE, FC, FLAGS

PROOF = Path(__file__).resolve().parent
STAGES = CACHE.parent / 'H38-53-literal2-stages.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    driver = PROOF / 'driver.f90'
    witness = json.loads((PROOF / 'receipt.json').read_text())
    assert sha(driver) == witness['driver_sha256']
    assert sha(CACHE / 'module_mp_thompson.F') == witness['source_sha256']
    tables = ['qr_acr_qg_mp38V1.dat', 'qr_acr_qsV2.dat', 'freezeH2O.dat', 'CCN_ACTIVATE.BIN']
    before = {name: (CACHE / name).stat() for name in tables}
    for name in tables:
        (out / name).symlink_to(CACHE / name)
    subprocess.run([FC, *FLAGS, '-I' + str(CACHE), '-c', str(driver), '-o', 'driver.o'], cwd=out, check=True, timeout=60)
    objects = ['thompson_debug.o', 'stubs.o', 'module_model_constants.o', 'module_mp_radar.o', 'libmassv.o']
    subprocess.run([FC, *FLAGS, '-o', 'oracle_debug', 'driver.o', *[str(CACHE / name) for name in objects]], cwd=out, check=True, timeout=60)
    result = {'source_sha256': witness['source_sha256'], 'driver_sha256': sha(driver), 'debug_object_sha256': sha(CACHE / 'thompson_debug.o'),
              'executable_sha256': sha(out / 'oracle_debug'), 'flags': FLAGS, 'candidate_stage_path': str(STAGES),
              'candidate_stage_sha256': sha(STAGES), 'cases': {}, 'JAX_imported': False}
    for batch, case in enumerate([10, 11]):
        ni = np.asarray(witness['cases'][str(case)]['target_input_ni'], np.float32)
        path = out / ('ni' + str(case) + '.txt')
        path.write_text('\n'.join(format(float(x), '.9e') for x in ni) + '\n')
        arrays = ['rho', 'ri', 'ni', 'qiten', 'niten', 'vtik', 'vtnik', 'qi1d', 'ni1d']
        scalar_names = ['rho(1)', 'qi1d(1)', 'ni1d(1)', 'qiten(1)', 'niten(1)', 'lami', 'ilami', 'xDi']
        commands = ['set pagination off', 'set confirm off', 'set debuginfod enabled off', 'set language fortran', 'python import gdb,json; stages={}']
        for phase, line in [('source', 3189), ('preSED', 3790), ('postSED', 3947), ('final_baseline', 4023), ('final_applied', 4029), ('final_slope', 4032), ('final_result', 4040)]:
            conditional = ' if k==1' if phase.startswith('final') else ''
            commands += ['break module_mp_thompson.F:' + str(line) + conditional, 'commands', 'silent']
            if phase.startswith('final'):
                code = 'stages[' + repr(phase) + ']={name:{"value":float(gdb.parse_and_eval(name)),"kind":str(gdb.parse_and_eval(name).type),"bytes":int(gdb.parse_and_eval(name).type.sizeof)} for name in ' + repr(scalar_names) + '}'
            else:
                code = 'stages[' + repr(phase) + ']={name:[float(gdb.parse_and_eval(name+"("+str(i)+")")) for i in range(1,41)] for name in ' + repr(arrays) + '}'
            commands += ['python ' + code, 'continue', 'end']
        capture = out / ('case' + str(case) + '_stages.json')
        commands += ['run ' + str(case) + ' ' + path.name, 'python open(' + repr(str(capture)) + ',"w").write(json.dumps(stages,indent=2))', 'quit']
        script = out / ('case' + str(case) + '.gdb')
        script.write_text('\n'.join(commands) + '\n')
        proc = subprocess.run(['gdb', '-q', '-batch', '-x', str(script), str(out / 'oracle_debug')], cwd=out,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
        (out / ('case' + str(case) + '.log')).write_text(proc.stdout)
        assert proc.returncode == 0, proc.stdout[-3000:]
        original = json.loads(capture.read_text())
        assert len(original) == 7, list(original)
        pairs = [('source', 'niten', '_wrf_mixed_phase_sources_niten'), ('preSED', 'ni', '_working_moments_returned_ni'),
                 ('postSED', 'niten', '_sedimentation_tendencies_Ni')]
        comparisons = []
        with np.load(STAGES, allow_pickle=False) as candidate:
            for phase, name, key in pairs:
                want = np.asarray(original[phase][name], np.float32)
                got = candidate[key][batch]
                different = np.flatnonzero(want.view(np.uint32) != got.view(np.uint32))
                comparisons.append({'original_stage': phase, 'original_field': name, 'candidate_key': key,
                                    'BYTE_count': 40-len(different), 'different_levels_1based': (different+1).tolist(),
                                    'level1_original': float(want[0]), 'level1_candidate': float(got[0])})
            for phase, name, key in [('final_applied', 'ni1d(1)', '_finish_aero_ni_raw'), ('final_applied', 'qi1d(1)', '_finish_aero_qi'),
                                     ('final_applied', 'rho(1)', '_finish_aero_rho'), ('final_slope', 'lami', '_finish_aero_lami'),
                                     ('final_slope', 'xDi', '_finish_aero_xdi'), ('final_result', 'ni1d(1)', '_finish_aero_Ni')]:
                entry = original[phase][name]
                want = entry['value']
                got = float(candidate[key][batch, 0])
                fmt = '>d' if entry['bytes'] == 8 else '>f'
                comparisons.append({'original_stage': phase, 'original_field': name, 'candidate_key': key,
                                    'original_kind': entry['kind'], 'original': want, 'candidate': got,
                                    'original_hex': struct.pack(fmt, want).hex(), 'candidate_hex': struct.pack(fmt, got).hex(), 'BYTE': want == got})
            baseline = candidate['_sedimentation_baseline_Ni'][batch]
        with np.load(PROOF / 'original_columns.npz', allow_pickle=False) as reference:
            incoming = reference[f'case{case}__NI_IN']
            assert np.array_equal(np.asarray(original['source']['ni1d'], np.float32).view(np.uint32), incoming.view(np.uint32))
            assert original['final_result']['ni1d(1)']['value'] == float(reference[f'case{case}__NI_OUT'][0])
        result['cases'][str(case)] = {'original': original, 'comparisons': comparisons,
                                     'source_baseline_input_BYTE_count': 40, 'final_level1_old_reference_BYTE': True,
                                     'negative_input_levels_1based': (np.flatnonzero(incoming < 0)+1).tolist(),
                                     'candidate_baseline_level1': float(baseline[0]),
                                     'cause': 'Original raw NI1D baseline remains negative while private working ni is floored. Native _clip_species_aero clamps incoming Ni to0 before HailContext.baseline/source tendency balance, changing source subtraction and later baseline+tendency cancellation rounding. Original3035-3053 uses raw NI1D in the balance.'}
    result['cached_tables_unchanged'] = all((before[name].st_ino, before[name].st_size, before[name].st_mtime_ns) ==
        ((CACHE / name).stat().st_ino, (CACHE / name).stat().st_size, (CACHE / name).stat().st_mtime_ns) for name in tables)
    assert result['cached_tables_unchanged'] and 'jax' not in sys.modules
    packed = {}
    for case, entry in result['cases'].items():
        for phase in ['source', 'preSED', 'postSED']:
            for name, value in entry['original'][phase].items():
                packed[f'case{case}__{phase}__{name}'] = np.asarray(value, np.float32)
        for phase in ['final_baseline', 'final_applied', 'final_slope', 'final_result']:
            for name, value in entry['original'][phase].items():
                packed[f'case{case}__{phase}__{name}'] = np.asarray(value['value'], np.float64 if value['bytes'] == 8 else np.float32)
    packed_path = out / 'original_threshold_number_stages.npz'
    np.savez_compressed(packed_path, **packed)
    result.update(packed_stage_sha256=sha(packed_path), packed_stage_arrays=len(packed))
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    for name in tables + ['driver.o', 'oracle_debug']:
        (out / name).unlink()
    print(json.dumps({case: entry['comparisons'] for case, entry in result['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
