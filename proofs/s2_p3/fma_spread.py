"""Independent original-WRF contraction and compiler-init sensitivity.

The pristine module is copied byte-for-byte. Only the external driver reads
frozen initial operands, so compiler variants perform identical scientific
work. Bounds are WRF-vs-WRF and never fitted to candidate errors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from generate_columns import FIELDS, driver


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--compiler', required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_p3.F')
    original = source.read_bytes()
    code = driver()
    marker = "open(newunit=u,file='frames.bin'"
    assert code.count(marker) == 1
    code = code.replace(marker, "open(newunit=u,file='initial.bin',access='stream',form='unformatted',status='old')\nread(u)x\nclose(u)\n" + marker)
    report = {'source_sha256': hashlib.sha256(original).hexdigest(),
              'operands': 'frozen original P305 frame0, identical across arms',
              'compiler': subprocess.check_output([args.compiler, '--version']).decode().splitlines()[0],
              'arms': {}, 'scores': {}}
    saved = {}
    for arm, contraction, init in [('literal', 'off', 'zero'), ('fma', 'fast', 'zero'), ('init_sensitivity', 'off', 'snan')]:
        root = args.out / arm
        root.mkdir(exist_ok=True)
        staged = root / 'module_mp_p3.F'
        staged.write_bytes(original)
        assert staged.read_bytes() == original
        (root / 'columns.f90').write_text(code)
        flags = ['-cpp', '-ffree-form', '-ffree-line-length-none', '-O2', '-march=native',
                 '-ffp-contract=' + contraction, '-fcheck=all', '-finit-real=' + init,
                 '-finit-integer=' + ('0' if init == 'zero' else '42'),
                 '-finit-logical=' + ('false' if init == 'zero' else 'true')]
        subprocess.run([args.compiler, *flags, 'module_mp_p3.F', 'columns.f90', '-o', 'columns'],
                       cwd=root, check=True, timeout=120)
        report['arms'][arm] = {'flags': flags, 'module_BYTE': True, 'schemes': {}}
        saved[arm] = {}
        for mp in (50, 51, 52, 53):
            folder = root / str(mp)
            folder.mkdir(exist_ok=True)
            with np.load(args.fixtures / f'columns_mp{mp}.npz') as ref:
                initial = np.stack([ref[name][..., 0] for name in FIELDS], axis=-1)[:, :, None, :]
            initial.ravel(order='F').astype(np.float32).tofile(folder / 'initial.bin')
            with (folder / 'stdout.log').open('w') as log:
                result = subprocess.run([str((root / 'columns').resolve()), str(source.parent.parent / 'run'), str(mp)],
                                        cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            entry = {'rc': result.returncode}
            report['arms'][arm]['schemes'][str(mp)] = entry
            if result.returncode:
                continue
            x = np.fromfile(folder / 'frames.bin', np.float32).reshape((8, 12, 1, len(FIELDS), 4), order='F')
            precip = np.fromfile(folder / 'precip.bin', np.float32).reshape((8, 1, 5, 4), order='F')
            data = {name: x[:, :, 0, i, :] for i, name in enumerate(FIELDS)}
            data.update({name: precip[:, 0, i, :] for i, name in enumerate(('rainnc', 'rainncv', 'sr', 'snownc', 'snowncv'))})
            entry['nonfinite'] = sum(int((~np.isfinite(value)).sum()) for value in data.values())
            saved[arm][mp] = data
            np.savez_compressed(args.out / f'{arm}_mp{mp}.npz', **data)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for mp in (50, 51, 52, 53):
        with np.load(args.fixtures / f'columns_mp{mp}.npz') as frozen:
            literal = saved['literal'][mp]
            fma = saved['fma'][mp]
            sensitivity = saved['init_sensitivity'].get(mp, {})
            scores = {}
            for name, value in literal.items():
                spread = np.abs(fma[name].astype(np.float64) - value.astype(np.float64))
                other = sensitivity.get(name)
                scores[name] = {'frozen_BYTE': value.tobytes() == frozen[name].tobytes(),
                                'fma_max_abs': float(np.nanmax(spread)),
                                'fma_different': int(np.count_nonzero(fma[name] != value)),
                                'init_BYTE': other is not None and other.tobytes() == value.tobytes()}
            report['scores'][str(mp)] = scores
    report['literal_all_BYTE'] = all(s['frozen_BYTE'] for mp in report['scores'].values() for s in mp.values())
    report['init_all_BYTE'] = all(s['init_BYTE'] for mp in report['scores'].values() for s in mp.values())
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'literal_all_BYTE': report['literal_all_BYTE'], 'init_all_BYTE': report['init_all_BYTE'],
                      'fma_spread': {mp: {n: s['fma_max_abs'] for n, s in scores.items() if s['fma_different']}
                                     for mp, scores in report['scores'].items()}}), flush=True)


if __name__ == '__main__':
    main()
