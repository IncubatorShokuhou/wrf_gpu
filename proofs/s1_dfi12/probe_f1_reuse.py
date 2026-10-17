"""Prove F1 changes only DFL host text, with six measured stage namelists exact.

Usage: python probe_f1_reuse.py IMMUTABLE_SOURCE OUTPUT_JSON
Minimal Times-only NetCDF inputs exercise the real writer; no weather simulation.
"""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

source, output = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(source / 'src'))
from gpuwrf.dfi import driver, driver12
from gpuwrf.dfi.control import dfi_control_from_namelist
from gpuwrf.io.namelist_check import _parse_wrf_namelist
from gpuwrf.io.netcdf_lock import Dataset

assert Path(driver12.__file__).resolve().is_relative_to(source.resolve())
reviewed = '3b2a8f12b'
def git(*args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
def original(path):
    return subprocess.check_output(['git', '-C', str(source), 'show', f'{reviewed}:{path}'], text=True)
def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
def canonical(node):
    return ast.dump(node, include_attributes=False)

shared = {}
for path in ('src/gpuwrf/dfi/driver.py', 'src/gpuwrf/dfi/filter.py', 'src/gpuwrf/dfi/control.py'):
    shared[path] = original(path) == (source / path).read_text()
assert all(shared.values()), shared
path = 'src/gpuwrf/dfi/driver12.py'
old, new = functions(original(path)), functions((source / path).read_text())
helpers = {k: canonical(old[k]) == canonical(new[k]) for k in old if k != 'run_dfi12'}
assert all(helpers.values()), helpers
run = new['run_dfi12']
selectors = [n for n in run.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'write_stage' for t in n.targets)]
assert len(selectors) == 1
assert ast.unparse(selectors[0].value) == '_write_dfl_stage_dir if ctl.dfi_opt == DFI_DFL else d._write_stage_dir'
run.body.remove(selectors[0])
calls = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'write_stage']
assert len(calls) == 2
for call in calls:
    call.func = ast.Attribute(value=ast.Name(id='d', ctx=ast.Load()), attr='_write_stage_dir', ctx=ast.Load())
assert canonical(run) == canonical(old['run_dfi12'])

lane = output.parent
stages = []
with tempfile.TemporaryDirectory(dir=lane, prefix='f1-host-') as temp:
    root = Path(temp)
    for case in ('opt1', 'dfl-default-short', 'dfl-endonly-short'):
        text = (lane / 'oracle' / case / 'namelist.input').read_text()
        nml = _parse_wrf_namelist(text)
        ctl = dfi_control_from_namelist(nml)
        src = root / case
        src.mkdir()
        (src / 'namelist.input').write_text(text)
        with Dataset(src / 'wrfinput_d01', 'w') as ds:
            ds.createDimension('Time', 1)
            ds.createDimension('DateStrLen', 19)
            ds.createVariable('Times', 'S1', ('Time', 'DateStrLen'))[0] = np.array(list('2023-01-15_00:00:00'), 'S1')
        for phase, when in (('forward', ctl.fwd_start), ('forecast', ctl.forecast_start)):
            edits = {**driver12._clock_edits(when), **driver12._dfl_diffusion(nml)}
            if phase == 'forward':
                edits.update(sst_update=0, dfi_opt=0)
            kwargs = dict(fields={}, when=when, namelist_edits=edits)
            old_stage = driver._write_stage_dir(src, root / f'{case}-old-{phase}', **kwargs)
            fixed = driver12._write_dfl_stage_dir(src, root / f'{case}-fixed-{phase}', **kwargs)
            a, b = (old_stage / 'namelist.input').read_bytes(), (fixed / 'namelist.input').read_bytes()
            assert a == b, (case, phase)
            stages.append(dict(case=case, phase=phase, byte_exact=True, namelist_bytes=len(a)))
result = dict(utc=datetime.now(timezone.utc).isoformat(), source_commit=git('rev-parse', 'HEAD'),
              src_tree=git('rev-parse', 'HEAD:src/gpuwrf'), reviewed_source=reviewed,
              shared_source_exact=shared, retained_driver12_helpers_exact=helpers,
              run_dfi12_exact_except_two_dfl_writer_calls=True, measured_dfl_stage_namelists=stages,
              numerical_reuse='Original-WRF prepass truth reused; this proves host stage files and retained numerical source only.',
              performance='Host namelist read/edit/write only; unchanged NetCDF data path, no added device operations or persistent arrays.')
output.write_text(json.dumps(result, indent=2) + '\n')
print('F1 SOURCE/HOST REUSE PASS', len(stages), 'stage files exact')
