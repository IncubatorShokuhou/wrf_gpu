"""Manager D control: exact fresh real.exe schema, source-resolved values.

Diagnostic only: provenance lives beside inputs so no attribute is added to
the files. Input-only derived values reuse the already measured native writer.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
from netCDF4 import Dataset

ROOT = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart')
OUT = ROOT / 'native_may04_00_real_schema'
FRESH = ROOT / 'may_pair/20260503_18z_a1'
NATIVE = ROOT / 'native_may04_00_newforcing'
SOURCE = ROOT / 'may_pair/previous_at_new_time'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def attrs(obj):
    return {n: {'dtype': np.asarray(obj.getncattr(n)).dtype.str,
                'shape': list(np.asarray(obj.getncattr(n)).shape),
                'bytes': np.asarray(obj.getncattr(n)).tobytes().hex()}
            for n in obj.ncattrs()}


def schema(ds):
    return {'format': ds.data_model, 'dimensions': {n: [len(d), d.isunlimited()] for n, d in ds.dimensions.items()},
            'attrs': attrs(ds), 'variables': {n: {'dtype': v.dtype.str, 'dims': list(v.dimensions), 'attrs': attrs(v)}
                                            for n, v in ds.variables.items()}}


def main():
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'QUIET'
    assert shutil.disk_usage(ROOT).free >= 200 * 2**30
    assert not OUT.exists()
    OUT.mkdir()
    start = time.monotonic()
    report = {'schema': 1, 'case': str(OUT), 'scope': 'D exact-real-schema ORIGINAL-WRF startup control',
              'cache': 'NONE', 'script_sha256': sha(__file__), 'domains': [], 'files': []}
    for i in (1, 2, 3):
        name = f'wrfinput_d{i:02}'
        target = OUT / name
        original = FRESH / name
        source = SOURCE / f'wrfout_d{i:02}_2026-05-04_00:00:00'
        shutil.copyfile(original, target)
        rows = {}
        with Dataset(original) as fresh, Dataset(source) as old, Dataset(NATIVE / name) as native, Dataset(target, 'r+') as out:
            for ds in (fresh, old, native, out):
                ds.set_auto_maskandscale(False)
                ds.set_auto_chartostring(False)
            before = schema(fresh)
            for n, v in out.variables.items():
                if n in ('XTIME', 'STEP_NUMBER'):
                    src, label = fresh, 'fresh issue clock'
                elif n in old.variables:
                    src, label = old, 'original source storage'
                else:
                    src, label = native, 'native input-only/derived value (WS13 provenance)'
                assert src[n].dimensions == v.dimensions and src[n].dtype == v.dtype, n
                assert src[n].shape == v.shape, n
                values = np.asarray(src[n][:])
                assert values.dtype.kind not in 'fiu' or np.isfinite(values).all(), n
                v[:] = values
                actual = np.asarray(v[:])
                assert actual.dtype == values.dtype and actual.tobytes() == values.tobytes(), n
                rows[n] = {'source': label, 'shape': list(values.shape), 'dtype': str(values.dtype),
                           'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
            out.sync()
            assert schema(out) == before, 'schema changed'
            after = schema(out)
        target.chmod(0o444)
        report['domains'].append({'domain': i, 'fresh_schema_sha256': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
            'output_schema_sha256': hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
            'variables': len(rows), 'source_variables': sum(r['source'] == 'original source storage' for r in rows.values()),
            'fields': rows, 'fresh_file_sha256': sha(original), 'source_file_sha256': sha(source)})
    for name in ['wrfbdy_d01', *(f'wrflowinp_d{i:02}' for i in (1, 2, 3)), 'namelist.input']:
        shutil.copyfile(NATIVE / name, OUT / name)
        assert sha(OUT / name) == sha(NATIVE / name)
        (OUT / name).chmod(0o444)
    for path in sorted(OUT.iterdir()):
        report['files'].append({'name': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    report.update(wall_s=time.monotonic()-start, status='PASS', outputs_not_executed=True)
    for tag, directory in [('A', FRESH), ('B', SOURCE), ('D', OUT)]:
        name = 'wrfout_d01_2026-05-04_00:00:00' if tag == 'B' else 'wrfinput_d01'
        with (OUT.parent / ('header_' + tag + '.txt')).open('x') as stream:
            subprocess.run(['ncdump', '-h', str(directory / name)], stdout=stream, check=True)
    receipt = Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/REAL_SCHEMA_D.json')
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'domains'}, indent=2))


if __name__ == '__main__':
    main()
