"""Stage verbatim WRF MMF routines and thin callers in an owned output dir.

Run under the lane CPU/QUIET/timeout wrapper. Output is disposable; pristine
sources and tables are never edited. The NPZ truth fixtures remain tracked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def routine(source, name):
    start = re.search(r'^[ \t]*SUBROUTINE\s+'+name+r'\s*\(', source, re.M | re.I).start()
    end = re.search(r'^\s*END\s+SUBROUTINE\s+'+name+r'\s*$', source[start:], re.M | re.I).end()
    return source[start:start+end]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pristine', type=Path, help='original WRF source root containing phys and run')
    p.add_argument('output', type=Path, help='new disposable lane-owned directory')
    p.add_argument('--fc', default='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit(3)
    args.output.mkdir(parents=True, exist_ok=False)
    src = args.pristine / 'phys/noahmp'
    lsm = (src/'src/module_sf_noahmplsm.F').read_text()
    driver = (src/'drivers/wrf/module_sf_noahmpdrv.F').read_text()
    groundwater = (src/'src/module_sf_noahmp_groundwater.F').read_text()
    tables = lsm[lsm.index('MODULE NOAHMP_TABLES'):]
    names = ['SOILWATER','SRT','SSTEP','ROSR12','WDFCND1','WDFCND2','SHALLOWWATERTABLE','INFIL','ZWTEQ']
    bodies = {n:routine(lsm,n) for n in names}
    start = lsm.index('  TYPE noahmp_parameters !')
    end = lsm.index('  END TYPE noahmp_parameters',start)+len('  END TYPE noahmp_parameters')
    columns = ('module mmf_column_original\nimplicit none\n'
        'integer,parameter::NSOIL=4,NSTAGE=8,MBAND=2,MSC=8,MVT=27\n'
        'integer::OPT_RUN=5,OPT_INF=1,OPT_TDRN=0\n'+lsm[start:end]+'\ncontains\n'
        +'\n'.join(bodies.values())+'\nend module\n')
    cold = ('module module_domain\n type domain\n integer :: unused=0\n end type\nend module\n'
        'module pristine_groundwater_init\ncontains\n'+routine(driver,'GROUNDWATER_INIT')
        +'\n'+routine(driver,'EQSMOISTURE')+'\nend module\n')
    here = Path(__file__).resolve().parent
    flags = ['-cpp','-DEM_CORE=1','-O0','-ffree-form','-ffree-line-length-none',
        '-fallow-argument-mismatch','-ffp-contract=off','-finit-real=zero',
        '-finit-integer=0','-finit-logical=false']
    layouts = {
        'grid': {'groundwater.F90':groundwater},
        'init': {'groundwater.F90':groundwater,'groundwater_init.F90':cold},
        'column': {'column_original.F90':columns},
    }
    for kind, sources in layouts.items():
        folder = args.output/kind
        folder.mkdir()
        (folder/'noahmp_tables.F90').write_text(tables)
        for name, text in sources.items(): (folder/name).write_text(text)
        for table in ('MPTABLE.TBL','SOILPARM.TBL','GENPARM.TBL'):
            (folder/table).symlink_to((args.pristine/'run'/table).resolve())
        wrapper = 'groundwater_driver.F90' if kind=='grid' else kind+'_driver.F90'
        shutil.copyfile(here/wrapper, folder/'driver.F90')
        paths = ['noahmp_tables.F90',*sources]
        if kind=='column':
            shutil.copyfile(here/'column_stubs.F90',folder/'stubs.F90')
            paths.append('stubs.F90')
        subprocess.run([args.fc,*flags,*paths,'driver.F90','-o','oracle.exe'],cwd=folder,
                       check=True,timeout=90)
    hashes = {n:hashlib.sha256(b.encode()).hexdigest() for n,b in bodies.items()}
    hashes.update({n:hashlib.sha256(routine(driver,n).encode()).hexdigest()
                   for n in ('GROUNDWATER_INIT','EQSMOISTURE')})
    receipt = {'bodies_verbatim':True,'routine_sha256':hashes,'flags':flags,
               'compiler':subprocess.check_output([args.fc,'--version'],text=True).splitlines()[0],
               'pristine':str(args.pristine.resolve()),'output':str(args.output.resolve())}
    (args.output/'SOURCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
