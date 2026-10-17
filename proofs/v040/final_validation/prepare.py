"""Prepare matched, original-WRF composite references; never opens a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
PRISTINE = Path('<USER_HOME>/src/wrf_pristine/WRF')
RETAINED = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual')
SLOTS = ('mp_physics', 'cu_physics', 'bl_pbl_physics', 'sf_sfclay_physics',
         'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configurations():
    # MP slots are mutually exclusive. These 26 rows are a lower bound, not a
    # claim that fifteen distinct programs cover twenty-six different MP codes.
    rows = [
        (8, 0, 5, 5, 4, 4, 4),
        (0, 0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 4, 1, 1),
        (2, 2, 2, 2, 4, 4, 2),
        (3, 3, 3, 3, 4, 4, 1),
        (4, 5, 7, 1, 4, 4, 4),
        (6, 6, 8, 1, 4, 4, 4),
        (9, 16, 11, 1, 4, 4, 4),
        (10, 14, 12, 1, 4, 4, 4),
        (13, 94, 99, 1, 4, 4, 4),
        (14, 95, 16, 1, 4, 4, 4),
        (16, 96, 17, 1, 4, 4, 4),
        (17, 0, 5, 5, 4, 4, 4),
        (18, 5, 5, 5, 4, 4, 4),
        (19, 99, 5, 5, 4, 4, 4),
        (21, 0, 5, 5, 4, 4, 4),
        (22, 0, 5, 5, 4, 4, 4),
        (24, 93, 5, 5, 4, 4, 4),
        (26, 0, 5, 5, 4, 4, 4),
        (28, 0, 5, 5, 4, 4, 4),
        (40, 0, 5, 5, 4, 4, 4),
        (50, 0, 5, 5, 4, 4, 4),
        (51, 0, 5, 5, 4, 4, 4),
        (52, 0, 5, 5, 4, 4, 4),
        (53, 0, 5, 5, 4, 4, 4),
        (97, 0, 0, 91, 4, 4, 4),
        (8, 0, 4, 4, 4, 4, 4),
        (8, 0, 10, 10, 4, 4, 4),
        (8, 0, 9, 5, 4, 3, 3),
        (8, 7, 9, 5, 4, 4, 4),
        (8, 10, 5, 5, 4, 4, 4),
        (8, 11, 5, 5, 4, 4, 4),
        (8, 0, 5, 5, 4, 5, 5),
        (8, 0, 5, 5, 3, 4, 4),
        (8, 0, 1, 1, 4, 4, 4),
        (8, 0, 8, 1, 4, 4, 4),
        (8, 1, 5, 5, 4, 4, 4),
        (8, 0, 0, 5, 4, 4, 4),
        (8, 0, 0, 5, 4, 4, 4),
        (0, 0, 0, 0, 0, 31, 0),
    ]
    out = []
    for i, values in enumerate(rows):
        name = f'c{i:02d}'
        row = dict(id=name, geography='Swiss', hours=24, domains=['d01'],
                   options=dict(zip(SLOTS, values)), extra={}, status='PLANNED')
        if i == 33:
            row['geography'] = 'Madrid-RUC'
        if i in (34, 35, 36):
            row['geography'] = 'Madrid-urban'
            row['extra'] = {'physics': {'sf_urban_physics': {34: 2, 35: 3, 36: 1}[i]}}
        if i == 37:
            row['extra'] = {'dynamics': {'diff_opt': 2, 'km_opt': 5}}
        if i == 38:
            row['extra'] = {'dynamics': {'diff_opt': 2, 'km_opt': 2}}
        if values[0] in (9, 17, 18, 19, 21, 22, 24, 26, 28, 50, 51, 52, 53) or i == 32:
            row['extra'].setdefault('physics', {}).update(use_mp_re=1)
        out.append(row)
    return out


def render(groups):
    def scalar(value):
        if isinstance(value, bool):
            return '.true.' if value else '.false.'
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)
    lines = []
    for group, entries in groups.items():
        lines.append('&' + group)
        for key, value in entries.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            lines.append(' ' + key + ' = ' + ', '.join(map(scalar, values)) + ',')
        lines.append('/\n')
    return '\n'.join(lines)


def private_executable(case, reference=PRISTINE):
    original = reference / 'main/wrf.exe'
    raw = original.read_bytes()
    result = raw
    patches = []
    for index, old in enumerate((b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt',
                                 b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt')):
        new = str(case / f'd{index}').encode()
        assert len(new) <= len(old) and result.count(old) == 1
        patches.append({'offset': result.index(old), 'old': old.decode(), 'new': new.decode()})
        result = result.replace(old, new.ljust(len(old), b' '))
    path = case / 'wrf.exe'
    path.write_bytes(result)
    path.chmod(0o755)
    restored = bytearray(result)
    for p in patches:
        start = p['offset']
        restored[start:start + len(p['old'])] = p['old'].encode()
    assert bytes(restored) == raw
    write_json(case / 'CPU_executable.json', {
        'original': str(original), 'original_sha256': sha(original),
        'private_sha256': sha(path), 'patches': patches,
        'arithmetic_bytes_unchanged': True,
        'configure_sha256': sha(reference / 'configure.wrf'),
        'mode': 'retained original serial WRF; only two private diagnostic OPEN literals changed',
    })


def prepare(source):
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.namelist_binding import namelist_honour_report
    from gpuwrf.io.scheme_catalog import _IMPLEMENTED
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = configurations()
    for row in rows:
        case = ROOT / row['id']
        assert not (case / 'cpu_run.json').exists(), 'Never rewrite a measured case'
        inp, cpu = case / 'input', case / 'cpu'
        inp.mkdir(parents=True, exist_ok=True)
        cpu.mkdir(exist_ok=True)
        if row['geography'] == 'Swiss':
            base = RETAINED / 'source_inputs/Swiss'
            template = source / 'examples/switzerland_d01/namelist.input'
        else:
            base = RETAINED / ('sf3_madrid' if row['geography'] == 'Madrid-RUC' else 'urban_madrid') / 'input'
            template = base / 'namelist.input'
        groups = parse_namelist(template)
        groups['physics'].update(row['options'])
        groups['physics']['sf_urban_physics'] = 0
        for group, values in row['extra'].items():
            groups.setdefault(group, {}).update(values)
        # Inputs, timeline, dt and soil dimensions are retained from real.exe.
        (inp / 'namelist.input').write_text(render(groups))
        errors, warnings = namelist_honour_report(groups, ('d01',), replay=False)
        row.update(admission_errors=errors, admission_warnings=warnings)
        row['status'] = 'ADMITTED' if not errors else 'REFUSED'
        for name in ('wrfinput_d01', 'wrfbdy_d01'):
            if not (inp / name).exists():
                (inp / name).symlink_to((base / name).resolve())
        private_executable(case)
        for item in (PRISTINE / 'run').iterdir():
            if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
                target = cpu / item.name
                if not target.exists():
                    target.symlink_to(item.resolve())
        for item in inp.iterdir():
            target = cpu / item.name
            if not target.exists():
                target.symlink_to(item.resolve())
        (cpu / 'wrf.exe').symlink_to(case / 'wrf.exe') if not (cpu / 'wrf.exe').exists() else None
        if row['options']['mp_physics'] in (8, 28):
            for table in ('qr_acr_qg_V4.dat', 'qr_acr_qsV2.dat'):
                target = cpu / table
                if not target.exists():
                    target.symlink_to(RETAINED / 'bl9/cpu' / table)
        row['input_sha256'] = {p.name: sha(p) for p in inp.iterdir() if p.is_file()}
        write_json(case / 'inputs.json', row)
    implemented = {key: sorted(values) for key, values in _IMPLEMENTED.items()}
    coverage = {key: sorted({r['options'][key] for r in rows if not r['admission_errors']})
                for key in SLOTS}
    missing = {key: sorted(set(values) - set(coverage.get(key, [])))
               for key, values in implemented.items()}
    matrix = dict(schema='v040-final-composites-v1', source_commit=subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        implemented=implemented, coverage=coverage, missing=missing, cases=rows,
        mp_lower_bound=len(implemented['mp_physics']),
        qualification='PREPARATION ONLY: no new GPU qualification or science PASS',
        api_only={'sf_surface_physics': [1, 2, 7]},
        excluded_cpu_only_radiation=[7, 99],
        pending_mp=[5, 7, 11, 15, 38, 55, 56])
    write_json(ROOT / 'MATRIX.json', matrix)
    print(json.dumps({'cases': len(rows), 'admitted': sum(not r['admission_errors'] for r in rows),
                      'missing': missing, 'refused': {r['id']: r['admission_errors'] for r in rows if r['admission_errors']}}, indent=2))


if __name__ == '__main__':
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
