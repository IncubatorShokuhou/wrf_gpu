"""Prepare full24h DFI and FDDA replay cases using original CPU-WRF data."""
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual')
SRC = ROOT / 'snapshot'
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')

def make(name, text, provenance):
    case = ROOT / name
    case.mkdir(exist_ok=True)
    inp = case / 'input'
    inp.mkdir(exist_ok=True)
    (inp / 'namelist.input').write_text(text)
    for key in ('wrfinput_d01', 'wrfbdy_d01'):
        (inp / key).symlink_to(SRC / 'examples/switzerland_d01' / key)
    cpu = case / 'cpu'
    cpu.mkdir(exist_ok=True)
    for item in (WRF / 'run').iterdir():
        if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
            (cpu / item.name).symlink_to(item.resolve())
    for item in inp.iterdir():
        (cpu / item.name).symlink_to(item.resolve())
    (cpu / 'wrf.exe').symlink_to(ROOT / 'pristine/wrf.exe')
    for item in (ROOT / 'bl9/cpu').glob('qr_acr*.dat'):
        (cpu / item.name).symlink_to(item)
    return case, inp, provenance

def seal(case, inp, provenance):
    result = {'schema': 'gpuqual-input-v1', 'config': case.name, 'hours': 24,
              'dt_s': 18, 'history_minutes': 60, 'provenance': provenance,
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inp.iterdir()}}
    (case / 'inputs.json').write_text(json.dumps(result, indent=2) + '\n')

def replace(text, key, value):
    text, n = re.subn(r'(?im)^(\s*' + key + r'\s*=)[^\n]*', lambda m: m[1] + f' {value},', text)
    assert n == 1, (key, n)
    return text

text = (SRC / 'tests/fixtures/o2_dfi_swiss_tdfi.namelist').read_text()
for key, value in {'run_days': 1, 'run_hours': 0, 'end_day': 16, 'end_hour': 0, 'history_interval': 60}.items():
    text = replace(text, key, value)
case, inp, provenance = make('dfi3', text, {
    'recipe': 'tracked o2_dfi Swiss oracle A; only forecast length/history cadence changed',
    'dfi_opt': 3, 'dfi_nfilter': 7, 'cutoff_s': 1200,
    'backward': '2023-01-15T00:00 to 2023-01-14T23:42 (60 steps)',
    'forward': '2023-01-14T23:51 to 2023-01-15T00:09 (60 steps)',
    'reference': 'same full original pristine WRF executable, 24h after filtering'})
seal(case, inp, provenance)

text = (SRC / 'examples/switzerland_d01/namelist.input').read_text()
text += """\n&fdda
 grid_fdda = 1,
 gfdda_inname = 'wrffdda_d<domain>',
 gfdda_interval_m = 60,
 gfdda_end_h = 24,
 io_form_gfdda = 2,
 guv = 0.0003,
 gt = 0.0003,
 gq = 0.0003,
 fgdt = 0,
/\n"""
case, inp, provenance = make('fdda1', text, {
    'analysis_values': 'ORIGINAL CPU-WRF RD11 hourly weather fields, 25 authentic frames',
    'analysis_layout': 'WRF auxinput10 OLD(t)/NEW(t+1h), 24 intervals',
    'scope': 'FDDA replay-analysis coupled qualification; no external-analysis or operational skill claim',
    'sources': '<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu',
    'formula': 'tracked proofs/o2_fdda/make_wrffdda.py literal REAL dry theta from THM/QVAPOR'})
spec = importlib.util.spec_from_file_location('gpuqual_fdda_writer', SRC / 'proofs/o2_fdda/make_wrffdda.py')
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)
writer.netCDF4 = SimpleNamespace(Dataset=Dataset)
sys.argv = ['make_wrffdda.py', provenance['sources'], 'd01', '2023-01-15_00:00:00', '60', '24', str(inp / 'wrffdda_d01')]
writer.main()
(case / 'cpu/wrffdda_d01').symlink_to(inp / 'wrffdda_d01')
seal(case, inp, provenance)
print('Prepared matched dfi3/fdda1 full24h inputs; FDDA targets are original CPU-WRF values')
