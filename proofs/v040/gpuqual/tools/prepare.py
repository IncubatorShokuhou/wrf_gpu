"""Swiss-only setup and private original executable; cities use onboard_cities.py."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual')
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
CONFIGS = {
    'bl9': {'bl_pbl_physics': 9},
    'cu5': {'cu_physics': 5},
    'cu93': {'cu_physics': 93},
    'km3': {'diff_opt': 2, 'km_opt': 3},
    'km2': {'diff_opt': 2, 'km_opt': 2},
    'ra3': {'ra_lw_physics': 3, 'ra_sw_physics': 3},
    'mp40': {'mp_physics': 40},
    'mp24': {'mp_physics': 24},
    'mp28': {'mp_physics': 28},
    'mp16': {'mp_physics': 16},
    'mp26': {'mp_physics': 26},
    'mp18': {'mp_physics': 18},
    'moist0': {'moist_adv_opt': 0},
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(source, name):
    case = ROOT / name
    case.mkdir(exist_ok=True)
    inp = case / 'input'
    inp.mkdir(exist_ok=True)
    namelist = (source / 'examples/switzerland_d01/namelist.input').read_text()
    for key, value in CONFIGS[name].items():
        namelist, n = re.subn(r'(?im)^(\s*' + key + r'\s*=)[^\n]*',
                             lambda m: m[1] + f' {value},', namelist)
        if n != 1:
            raise ValueError(f'{key}: cannot safely edit {n} occurrences; config needs explicit setup')
    (inp / 'namelist.input').write_text(namelist)
    for filename in ('wrfinput_d01', 'wrfbdy_d01'):
        (inp / filename).symlink_to(source / 'examples/switzerland_d01' / filename)
    cpu = case / 'cpu'
    cpu.mkdir(exist_ok=True)
    for item in (WRF / 'run').iterdir():
        if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
            (cpu / item.name).symlink_to(item.resolve())
    for item in inp.iterdir():
        (cpu / item.name).symlink_to(item.resolve())
    (cpu / 'wrf.exe').symlink_to(ROOT / 'pristine/wrf.exe')
    result = {'schema': 'gpuqual-input-v1', 'config': name, 'edits': CONFIGS[name],
              'source': str(source), 'hours': 24, 'dt_s': 18, 'history_minutes': 60,
              'input_sha256': {p.name: sha(p) for p in inp.iterdir()},
              'cpu_mode': 'pristine SERIAL on core29; no MPI throughput claim',
              'special_setup': 'none; single option plus WRF-required scheme controls'}
    (case / 'inputs.json').write_text(json.dumps(result, indent=2) + '\n')

def private_wrf():
    out = ROOT / 'pristine'
    out.mkdir(exist_ok=True)
    raw = (WRF / 'main/wrf.exe').read_bytes()
    result = raw
    patches = []
    # Preserve Fortran string lengths; OPEN trims trailing blank padding.
    for index, old in enumerate((b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt',
                                 b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt')):
        new = str(out / f'd{index}.txt').encode()
        if len(new) > len(old) or old not in result:
            raise ValueError('Unexpected executable dump pathname')
        padded = new.ljust(len(old), b' ')
        offsets = [m.start() for m in re.finditer(re.escape(old), result)]
        result = result.replace(old, padded)
        patches.append({'old': old.decode(), 'new': padded.decode(), 'offsets': offsets})
    path = out / 'wrf.exe'
    path.write_bytes(result)
    path.chmod(0o755)
    (out / 'receipt.json').write_text(json.dumps({
        'original': str(WRF / 'main/wrf.exe'), 'original_sha256': hashlib.sha256(raw).hexdigest(),
        'private_sha256': sha(path), 'equal_length': len(raw) == len(result),
        'only_changes': 'two instrumented OPEN path literals, arithmetic/source unchanged',
        'patches': patches, 'configure_sha256': sha(WRF / 'configure.wrf'),
    }, indent=2) + '\n')

if __name__ == '__main__':
    private_wrf()
    for name in sys.argv[2:]:
        prepare(Path(sys.argv[1]).resolve(), name)
