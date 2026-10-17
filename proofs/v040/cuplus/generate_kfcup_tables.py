"""Generate the production CuP REAL lookup asset from pristine KF_LUTAB.

Only the exact original lookup routine is extracted; the wrapper exposes
its arrays without changing any arithmetic. No previously ported table is
used as truth. Requires the lane CPU affinity, timeout, and QUIET guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from generate import extract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    import numpy as np

    wrf = Path('<USER_HOME>/src/wrf_pristine/WRF')
    original = wrf/'phys/module_cu_kfcup.F'
    source, bounds = extract(original.read_text(), 'kf_lutab')
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    (build/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(build/'tmp')
    (build/'lookup.f90').write_text('''module original_cup_lookup
  implicit none
  integer,parameter :: KFNT=250,KFNP=220
  real :: TTAB(KFNT,KFNP),QSTAB(KFNT,KFNP),THE0K(KFNP),ALU(200),RDPR,RDTHK,PLUTOP
contains
''' + source + '\nend module\n')
    shutil.copyfile(wrf/'share/module_model_constants.F', build/'module_model_constants.F')
    (build/'driver.f90').write_text('''program dump_tables
  use original_cup_lookup
  use module_model_constants,only: SVP1,SVP2,SVP3,SVPT0
  implicit none
  call kf_lutab(SVP1,SVP2,SVP3,SVPT0)
  open(71,file='tables.bin',form='unformatted',access='stream',status='replace')
  write(71) TTAB,QSTAB,THE0K,ALU,RDPR,RDTHK,PLUTOP
  close(71)
end program
''')
    compiler = shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags = ['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1']
    subprocess.run([compiler,*flags,'module_model_constants.F','lookup.f90','driver.f90','-o','dump'],
                   cwd=build, check=True, timeout=120)
    subprocess.run([str(build/'dump')], cwd=build, check=True, timeout=30)
    raw = np.fromfile(build/'tables.bin', '<f4')
    assert len(raw) == 110423 and np.isfinite(raw).all(), len(raw)
    data = dict(TTAB=raw[:55000].reshape((250,220), order='F'),
                QSTAB=raw[55000:110000].reshape((250,220), order='F'),
                THE0K=raw[110000:110220], ALU=raw[110220:110420],
                RDPR=raw[110420],RDTHK=raw[110421],PLUTOP=raw[110422])
    args.asset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.asset, **data)
    record = dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                  extracted_lines=bounds, extracted_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  constants_sha256=hashlib.sha256((build/'module_model_constants.F').read_bytes()).hexdigest(),
                  asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),
                  raw_sha256=hashlib.sha256((build/'tables.bin').read_bytes()).hexdigest(),
                  dtype='WRF REAL little-endian float32', elements=len(raw),
                  compiler=compiler, flags=flags, scope='CuP only; existing cu1 asset unchanged')
    args.receipt.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
