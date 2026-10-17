"""Build an unmodified, lane-local WRF TEMF surface oracle; no JAX needed.

Run with JAX_PLATFORMS=cpu and the lane CPU/QUIET guard. The source is copied
into the supplied artifact directory. It never modifies the pristine tree.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

INPUTS = ('u v theta qv pressure exner rho z terrain psfc roughness mavail tsk coriolis hfx te_temf hd_temf wm_temf').split()
OUTPUTS = ('te_temf wm_temf ust flhc flqc exch_temf chs hfx qfx lh qsfc u10 v10 t2 th2 q2 chs2 cqs2 qgh cpm').split()
HERE = Path(__file__).resolve().parent
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'


def columns():
    rows, names = [], []
    # Independent structured profiles: both Ri signs, its cap, zero shear,
    # incoming heat-flux signs, energy and WM floors, wet/dry, terrain offset.
    for stability, delta in [('unstable', -4.), ('neutral', 0.), ('stable', 4.), ('strong_stable', 35.)]:
        for wind in [0., .01, 2., 12.]:
            for wet in [False, True]:
                exner = np.float32(.97)
                tsk = np.float32(290.)
                terrain = 750. if wet else 0.
                vals = [wind, wind*.25, tsk/exner+delta, .006, 90000., exner, 1.08,
                        terrain+30., terrain, 92000., .001 if wet else .12,
                        1. if wet else .35, tsk, -8e-5 if wet else 0.,
                        150. if delta < 0 else -30., 0. if wind < 1 else .6,
                        800., 0. if wind < 1 else .2]
                names.append(f'{stability}_wind{wind}_{"water" if wet else "land"}')
                rows.append(dict(zip(INPUTS, map(float, np.asarray(vals, np.float32)))))
    return names, rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    build = Path(sys.argv[1]).resolve()
    build.mkdir(parents=True, exist_ok=False)
    sources = [WRF/'phys/module_sf_temfsfclay.F', WRF/'share/module_model_constants.F', HERE/'surface_driver.f90']
    for p in sources:
        shutil.copy2(p, build/p.name)
    cmd = [FC, '-O2', '-ffp-contract=off', '-ffree-form', '-ffree-line-length-none', '-cpp', '-DEM_CORE=1',
           '-fcheck=all', '-finit-real=snan', 'module_model_constants.F',
           'module_sf_temfsfclay.F', 'surface_driver.f90', '-o', 'oracle.exe']
    subprocess.run(cmd, cwd=build, check=True, timeout=60)
    names, rows = columns()
    inp = f'{len(rows)}\n'+'\n'.join(' '.join(str(r[k]) for k in INPUTS) for r in rows)+'\n'
    (build/'columns.txt').write_text(inp)
    result = subprocess.run([str(build/'oracle.exe')], input=inp, text=True, capture_output=True, check=True, timeout=30)
    (build/'raw.txt').write_text(result.stdout)
    recs = []
    carried = [r.copy() for r in rows]
    for line in result.stdout.splitlines():
        words = line.split()
        if len(words) != 22:
            continue  # WRF prints its zero-shear notice to stdout.
        step, idx = map(int, words[:2])
        out = dict(zip(OUTPUTS, map(float, words[2:])))
        assert all(np.isfinite(list(out.values())))
        recs.append(dict(name=names[idx-1], step=step, inputs=carried[idx-1].copy(), outputs=out))
        carried[idx-1].update({k: out[k] for k in ('te_temf', 'wm_temf', 'hfx')})
    assert len(recs) == 3*len(rows)
    data = dict(source='WRF v4.7.1 phys/module_sf_temfsfclay.F (unmodified)',
                caller='phys/module_surface_driver.F:2521-2533; full 3-D temfsfclay, lowest mass level; WRF REAL constants',
                precision='float32; no DOUBLE islands', gate='8 binary32 ULP of each output field scale per record; reject NaN/Inf',
                compile_command=cmd, sources={str(p):digest(p) for p in sources},
                compiler=subprocess.check_output([FC, '--version'], text=True).splitlines()[0], records=recs)
    path = build/'surface_oracle.json'
    path.write_text(json.dumps(data, indent=2)+'\n')
    print(f'{len(recs)} records, {len(OUTPUTS)*len(recs)} outputs; {path}; SHA256 {digest(path)}')


if __name__ == '__main__':
    main()
