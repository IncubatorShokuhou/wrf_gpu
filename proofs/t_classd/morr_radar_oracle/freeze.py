"""Freeze the pristine refl10cm_hm oracle outputs into fixtures/oracle.npz (+ copy inputs.npz).

usage: python freeze.py <work_dir_with_build_and_inputs> <fixture_dir>
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

work, fixture = Path(sys.argv[1]), Path(sys.argv[2])
fixture.mkdir(parents=True, exist_ok=True)
inputs = np.load(work / 'inputs.npz')
ncol, _, nz = inputs['columns'].shape
out, manifest = {}, {'arms': {}}
for arm in ('fp32_morr', 'fp32_morr_aero', 'fp64_morr', 'fp64_morr_aero'):
    text = (work / 'build' / arm / 'out.txt').read_text().splitlines()
    dbz = np.array([float(l.split()[4]) for l in text if l.startswith('DBZ')]).reshape(ncol, nz)
    bits = np.array([int(l.split()[3], 16) for l in text if l.startswith('DBZ')], np.uint64).reshape(ncol, nz)
    if arm.startswith('fp32'):
        assert np.array_equal(dbz.astype(np.float32).view(np.uint32), bits.astype(np.uint32)), arm
        out[arm] = dbz.astype(np.float32)
    else:
        assert np.array_equal(dbz.view(np.uint64), bits), arm
        out[arm] = dbz
    consts = {l.split()[0]: [float(x) for x in l.split()[1:]] for l in text if not l.startswith('DBZ')}
    manifest['arms'][arm] = dict(constants=consts,
                                 sources=(work / 'build' / arm / 'SOURCES.sha256').read_text().split('\n'))
np.savez_compressed(fixture / 'oracle.npz', **out)
shutil.copy(work / 'inputs.npz', fixture / 'inputs.npz')
manifest['inputs_sha256'] = hashlib.sha256((fixture / 'inputs.npz').read_bytes()).hexdigest()
manifest['oracle_sha256'] = hashlib.sha256((fixture / 'oracle.npz').read_bytes()).hexdigest()
(fixture / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(ncol, nz, {k: int((v > -35).sum()) for k, v in out.items()})
