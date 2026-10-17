"""Freeze exact WRF QNSE initializer tables; staged sources remain unmodified."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wrf',type=Path,required=True)
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit('QUIET')
    args.build.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in ('share/module_model_constants.F','phys/module_sf_qnsesfc.F'):
        source=args.wrf/name
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source,args.build/source.name)
    driver=Path(__file__).with_name('surface_tables.f90')
    shutil.copyfile(driver,args.build/driver.name)
    (args.build/'module_dm.f90').write_text('module module_dm\ncontains\nsubroutine wrf_dm_maxval(a,b,c)\ninteger::a,b,c\nstop 98\nend subroutine\nend module\n')
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O0','-ffree-form','-ffree-line-length-none','-cpp','-DNMM_CORE=0','-ffp-contract=off','-fcheck=all']
    arrays={}
    for mode,dtype,extra in [('fp32','<f4',[]),('fp64','<f8',['-fdefault-real-8','-fdefault-double-8'])]:
        subprocess.run([fc,*flags,*extra,'module_model_constants.F','module_dm.f90','module_sf_qnsesfc.F','surface_tables.f90','-o','tables'],cwd=args.build,check=True,timeout=90)
        subprocess.run([str(args.build/'tables')],cwd=args.build,check=True,timeout=10)
        values=np.fromfile(args.build/'tables.bin',dtype=dtype)
        assert values.shape==(20004,) and np.isfinite(values).all()
        arrays.update({mode+'_psim':values[:10001],mode+'_psih':values[10001:20002],mode+'_meta':values[20002:]})
    np.savez_compressed(args.output,**arrays)
    args.output.with_suffix('.json').write_text(json.dumps({'source_sha256':hashes,'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),'flags':flags,'asset_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'compiler':subprocess.check_output([fc,'--version'],text=True).splitlines()[0]},indent=2)+'\n')


if __name__=='__main__':main()
