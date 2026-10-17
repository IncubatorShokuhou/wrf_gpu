"""Stage pristine WRF locally, build BEP/BEM CPU oracles, save small fixtures.

Run with JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 28 nice -n 19
timeout 600 python build_oracle.py --build <lane artifact directory>.
No existing WRF build, output, or reference file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
SOURCES = ('share/module_model_constants.F', 'frame/module_wrf_error.F',
           'phys/module_sf_urban.F', 'phys/module_bep_bem_helper.F',
           'phys/module_sf_bem.F', 'phys/module_sf_bep.F', 'phys/module_sf_bep_bem.F')


def guard():
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit('QUIET')
    assert os.environ['JAX_PLATFORMS'] == 'cpu'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(src, start, end):
    text = src.read_text()
    a = re.search(r'^\s*SUBROUTINE ' + start + r'\b', text, re.I | re.M).start()
    b = re.search(r'^\s*END SUBROUTINE ' + end + r'\b[^\n]*', text[a:], re.I | re.M).end() + a
    return text[a:b] + '\n'


def parse_dump(src, dst):
    """Reject duplicate or missing records; absent step-0 diagnostics stay absent."""
    records = {}
    names = {}
    declared_regimes = None
    declared_steps = {}
    for line in src.read_text().splitlines():
        if line.startswith('NREG='):
            declared_regimes = int(line.split('=',1)[1])
        m = re.fullmatch(r'NSTEPS\[(\d+)\]=(\d+)', line)
        if m:
            assert int(m[1]) not in declared_steps
            declared_steps[int(m[1])] = int(m[2])
        m = re.fullmatch(r'REGIME_NAME\[(\d+)\]=(.*)', line)
        if m:
            assert int(m[1]) not in names
            names[int(m[1])] = m[2]
        m = re.fullmatch(r'(\w+)\[(\d+)\]\[(\d+)\]=\s*(\S+)\s*', line)
        if not m:
            continue
        key, r, s, value = m.groups()
        index = (int(r), int(s))
        assert index not in records.setdefault(key, {}), (key, index)
        records[key][index] = float(value)
    assert declared_regimes and records
    assert sorted(names) == list(range(1,declared_regimes+1)), 'early Fortran STOP: incomplete regimes'
    assert sorted(declared_steps) == sorted(names), 'missing step declarations'
    out = {'regime_names': np.array(list(names.values()))}
    for key, values in records.items():
        steps = sorted({s for r, s in values})
        expected = ({(r,0) for r in names} if key.startswith('IN_') else
                    {(r,s) for r in names for s in range(0 if 0 in steps else 1,declared_steps[r]+1)})
        assert set(values) == expected, ('incomplete step coverage',key)
        assert len({declared_steps[r] for r in names}) == 1
        arr = np.array([[values[(r, s)] for s in steps] for r in names])
        assert np.isfinite(arr).all(), key
        out[key] = arr
    groups = {}
    for key in list(out):
        m = re.fullmatch(r'(.+)_(\d+)', key)
        if m:
            groups.setdefault(m[1], {})[int(m[2])] = out.pop(key)
    for key, values in groups.items():
        assert sorted(values) == list(range(1, len(values)+1)), key
        out[key] = np.stack([values[k] for k in sorted(values)], axis=-1)
    np.savez_compressed(dst, **out)
    return {k: list(v.shape) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--save', type=Path, default=HERE/'savepoints')
    args = parser.parse_args()
    guard()
    args.build.mkdir(parents=True, exist_ok=True)
    args.save.mkdir(parents=True, exist_ok=True)
    fc = Path('<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    flags = ['-O2', '-ffree-form', '-ffree-line-length-none', '-cpp', '-DEM_CORE=1',
             '-fallow-argument-mismatch', '-finit-integer=0', '-fcheck=bounds']
    noah = extract(WRF/'phys/module_sf_noahmpdrv.F', 'noahmp_urban', 'noahmp_urban')
    cal = extract(WRF/'phys/module_ra_gfdleta.F', 'CAL_MON_DAY', 'CAL_MON_DAY')
    source_hashes = {s: sha(WRF/s) for s in SOURCES}
    source_hashes['phys/module_sf_noahmpdrv.F:noahmp_urban'] = hashlib.sha256(noah.encode()).hexdigest()
    source_hashes['phys/module_ra_gfdleta.F:CAL_MON_DAY'] = hashlib.sha256(cal.encode()).hexdigest()
    manifest = {'sources': source_hashes, 'compiler': subprocess.check_output([str(fc), '--version'], text=True).splitlines()[0],
                'flags': flags, 'kind_flags': {'fp64': ['-fdefault-real-8', '-fdefault-double-8'], 'fp32': []},
                'caller': 'verbatim Noah-MP noahmp_urban; surface-driver BEP/BEM diagnostic overwrite',
                'scope': 'constructed vertical profiles above tracked WN3 lowest-level forcings; fixed rural LSM and sfclay inputs; no PBL/forecast qualification',
                'fixtures': {}}
    for precision in ('fp64', 'fp32'):
        build = args.build/precision
        build.mkdir(exist_ok=True)
        for src in SOURCES:
            guard()
            shutil.copyfile(WRF/src, build/Path(src).name)
            assert sha(build/Path(src).name) == source_hashes[src]
        # Non-physical serial messaging and category-id dependencies only.
        stubs = (HERE.parent/'urban_slucm'/'build_and_run.sh').read_text()
        for nm, marker in [('noahmp_tables_stub.f90', 'noahmp_tables_stub.f90'), ('_wrfstubs.f90', '_wrfstubs.f90')]:
            content = stubs.split('"${STAGE}/'+marker+'" <<\'EOF\'\n', 1)[1].split('\nEOF', 1)[0]
            (build/nm).write_text(content+'\n')
        (build/'noahmp_urban_extract.F').write_text('MODULE noahmp_urban_extract\nCONTAINS\n'+noah+'END MODULE\n')
        (build/'module_ra_gfdleta_extract.F').write_text('MODULE module_ra_gfdleta\nCONTAINS\n'+cal+'END MODULE\n')
        pflags = flags+manifest['kind_flags'][precision]
        files = [Path(s).name for s in SOURCES]
        files[2:2] = ['_wrfstubs.f90']
        files += ['noahmp_tables_stub.f90', 'module_ra_gfdleta_extract.F', 'noahmp_urban_extract.F']
        for src in files:
            guard()
            subprocess.run([str(fc), *pflags, '-c', src], cwd=build, check=True, timeout=90)
        objects = [str(Path(s).with_suffix('.o')) for s in files]
        for scheme in (2, 3):
            guard()
            scheme_flags = ['-DWITH_BEM'] if scheme == 3 else []
            subprocess.run([str(fc), *pflags, *scheme_flags, '-o', f'oracle{scheme}', str(HERE/'bep_bem_oracle.F90'), *objects],
                           cwd=build, check=True, timeout=90)
            for table in ('ac_on', 'ac_off'):
                if scheme == 2 and table == 'ac_off':
                    continue
                run = build/f'run{scheme}_{table}'
                run.mkdir(exist_ok=True)
                tbl = (WRF/'run/URBPARM.TBL').read_text()
                if table == 'ac_off':
                    tbl = re.sub(r'^SW_COND:.*$', 'SW_COND: 0, 0, 0', tbl, flags=re.M)
                (run/'URBPARM.TBL').write_text(tbl)
                raw = run/'dump.txt'
                with raw.open('w') as stdout, (run/'stderr.txt').open('w') as stderr:
                    subprocess.run([str(build.resolve()/f'oracle{scheme}'), precision, str(HERE/'regimes.txt')],
                                   cwd=run, stdout=stdout, stderr=stderr, check=True, timeout=60)
                target = args.save/f'urban{scheme}_{precision}_{table}.npz'
                shapes = parse_dump(raw, target)
                manifest['fixtures'][target.name] = {'sha256': sha(target), 'fields': len(shapes), 'shapes': shapes,
                                                    'table_sha256': sha(run/'URBPARM.TBL')}
        guard()
        subprocess.run([str(fc), *pflags, '-o', 'components', str(HERE/'components_oracle.f90'), *objects], cwd=build, check=True, timeout=90)
        with (build/'components.txt').open('w') as output:
            subprocess.run([str(build.resolve()/'components')], cwd=build, stdout=output, check=True, timeout=60)
        values = np.loadtxt(build/'components.txt')
        assert np.isfinite(values).all()
        np.savez_compressed(args.save/f'components_{precision}.npz', values=values)
    manifest['inputs'] = {p.name: sha(p) for p in (HERE/'bep_bem_oracle.F90', HERE/'components_oracle.f90', HERE/'regimes.txt', Path(__file__))}
    manifest['table_sha256'] = sha(WRF/'run/URBPARM.TBL')
    (args.save/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'fixtures': len(manifest['fixtures']), 'manifest': str(args.save/'manifest.json')}))


if __name__ == '__main__':
    main()
