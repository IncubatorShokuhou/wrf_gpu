"""Prepare c07sd1 = admitted Swiss composite c07 (bl=11 Shin-Hong, mp9, cu16, RRTMG) with
shinhong_tke_diag=1 — CPU only, no JAX (lane w41-shinhongtke).

Only the namelist changes (one &physics entry); wrfinput/wrfbdy are symlinks to the v-prep c07
inputs (Swiss, not ALISIOS data). The cpu dir gets the pristine serial WRF 4.7.1 as a private
copy with only the two E104 debug OPEN literals redirected (w41-topowind private_executable).
usage: python prepare_c07sd1.py <cases_root>
"""
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
argv = sys.argv; sys.argv = [argv[0], str(ROOT)]
spec = importlib.util.spec_from_file_location('tw_prepare', HERE.parent / 'w41_topowind' / 'prepare_cases.py')
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)
sys.argv = argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE = tw.VPREP / 'c07' / 'input'
case = ROOT / 'c07sd1'
inp, cpu = case / 'input', case / 'cpu'
assert not list(cpu.glob('wrfout_*')), 'never overwrite reference frames'
inp.mkdir(parents=True, exist_ok=True); cpu.mkdir(exist_ok=True)
text = (BASE / 'namelist.input').read_text()
assert 'shinhong_tke_diag' not in text.lower()
new = tw.set_physics(text, 'shinhong_tke_diag', [1])
assert [l for l in new.splitlines() if l not in text.splitlines()] == [' shinhong_tke_diag = 1,']
(inp / 'namelist.input').write_text(new)
record = dict(case='c07sd1', base='c07', namelist_sha256=sha(inp / 'namelist.input'),
              base_namelist_sha256=sha(BASE / 'namelist.input'), namelist_delta=' shinhong_tke_diag = 1,', inputs={})
for f in ('wrfinput_d01', 'wrfbdy_d01'):
    target = inp / f
    if not target.is_symlink():
        target.symlink_to((BASE / f).resolve())
    record['inputs'][f] = dict(source=str((BASE / f).resolve()), sha256=sha(BASE / f), symlink=True)
for item in (tw.PRISTINE / 'run').iterdir():
    if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
        target = cpu / item.name
        if not target.exists():
            target.symlink_to(item.resolve())
for item in inp.iterdir():
    target = cpu / item.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(item.resolve() if item.is_symlink() else item)
target = cpu / 'MPTABLE.TBL'
if not target.exists():
    target.symlink_to(tw.PRISTINE / 'phys/noahmp/parameters/MPTABLE.TBL')
record['executable'] = tw.private_executable(case)
(case / 'PREPARE.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record, indent=1))
