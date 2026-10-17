"""Keep exact scanned executable PINs, then delete only case-owned caches."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
root = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
name = sys.argv[1]
assert re.fullmatch(r'c\d\d', name) or name == 'WN3-smoke'
case = root / name
assert case.resolve() == case
cache, temporary = case / 'cache', case / 'tmp'
spill = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms') / name
cache_physical, temporary_physical = cache.resolve(), temporary.resolve()
assert cache_physical in (cache, spill/'cache')
assert temporary_physical in (temporary, spill/'tmp')
gate_name=sys.argv[2] if len(sys.argv)>2 else 'e41'
assert gate_name in ('e41','e41_scanned_without_finalproof')
gate_dir=case/gate_name
assert gate_dir.resolve() in (gate_dir, spill/gate_dir.name)
gate = json.loads((gate_dir / 'wave_gate.json').read_text())
assert gate['coverage']['status'] == 'PASS', 'Incomplete scanner: retain its inputs'
configuration = case / ('PLAN.json' if name == 'WN3-smoke' else 'inputs.json')
assert all((case / f'D6_{d}.json').exists() for d in json.loads(configuration.read_text())['domains'])
pins = gate_dir / 'pins'
pins.mkdir(exist_ok=True)
rows = []
for i, entry in enumerate(gate['entries']):
    original = Path(entry['path'])
    assert original.resolve().is_relative_to(cache_physical)
    target = pins / f'{i}_{original.name}'
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    shutil.copyfile(original, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    rows.append(dict(original_cache_path=str(original), pin=str(target), sha256=digest, bytes=target.stat().st_size))
(gate_dir / 'PIN_MAP.json').write_text(json.dumps(dict(pins=rows, e41_gate=gate['e41_gate'],
    final_run_proof_available=bool(gate['coverage'].get('run_proofs')),
    scope='every actual scanned stepping executable, same native/HLO as full gate',
    original_cache_deleted=True, deleted_cache_physical=str(cache_physical)), indent=2) + '\n')
shutil.rmtree(cache_physical)
if cache.is_symlink():cache.unlink()
if temporary.exists():
    shutil.rmtree(temporary_physical)
    if temporary.is_symlink():temporary.unlink()
print(json.dumps(dict(case=name, pins=len(rows), retained_bytes=sum(r['bytes'] for r in rows))))
