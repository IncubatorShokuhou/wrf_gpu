"""Stage pristine WRF BEP/BEM sources in a lane dir and build the PV/green-roof oracle.

Same staging/flags as proofs/v040/urban_bep_bem/build_oracle.py (unmodified pristine modules,
non-physical stubs, verbatim noahmp_urban/CAL_MON_DAY extracts); only the driver differs
(pvgr_oracle.F90: extra history dumps + SMOIS_URB seed). BEM scheme (3) only.

JAX_PLATFORMS=cpu taskset -c <core> nice -n 19 timeout 900 python proofs/w41_writergaps/build_oracle.py --build <lane dir>
then run_oracle.py --build <lane dir> --save proofs/w41_writergaps/savepoints.
--relink re-links only the driver against an existing build's objects (staged sources sha-checked).
"""
import argparse, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
V040=HERE.parent/'v040/urban_bep_bem'
sys.path.insert(0,str(V040))
import build_oracle as base  # noqa: E402


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--relink',action='store_true')
    a=p.parse_args()
    base.guard()
    fc=Path('<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1','-fallow-argument-mismatch','-finit-integer=0','-fcheck=bounds']
    noah=base.extract(base.WRF/'phys/module_sf_noahmpdrv.F','noahmp_urban','noahmp_urban')
    cal=base.extract(base.WRF/'phys/module_ra_gfdleta.F','CAL_MON_DAY','CAL_MON_DAY')
    record={'sources':{s:base.sha(base.WRF/s) for s in base.SOURCES},'flags':flags,
            'compiler':subprocess.check_output([str(fc),'--version'],text=True).splitlines()[0],
            'driver_sha256':base.sha(HERE/'pvgr_oracle.F90'),'kind_flags':{'fp64':['-fdefault-real-8','-fdefault-double-8'],'fp32':[]}}
    if a.relink:
        previous=json.loads((a.build/'build_record.json').read_text())
        assert {k:previous[k] for k in ('sources','flags','compiler','kind_flags')}=={k:record[k] for k in ('sources','flags','compiler','kind_flags')}
        # Objects stay those of the original pristine compile; chained relinks keep pointing at it.
        record['objects_from']=previous.get('objects_from') or {k:previous[k] for k in previous if k.endswith('_oracle_sha256')}|{'driver_sha256':previous['driver_sha256']}
    stubs=(V040.parent/'urban_slucm'/'build_and_run.sh').read_text()
    for precision in ('fp64','fp32'):
        build=a.build/precision;build.mkdir(parents=True,exist_ok=True)
        for src in base.SOURCES:
            if not a.relink:
                base.guard();(build/Path(src).name).write_bytes((base.WRF/src).read_bytes())
            assert base.sha(build/Path(src).name)==record['sources'][src]
        for nm in ('noahmp_tables_stub.f90','_wrfstubs.f90'):
            (build/nm).write_text(stubs.split('"${STAGE}/'+nm+'" <<\'EOF\'\n',1)[1].split('\nEOF',1)[0]+'\n')
        (build/'noahmp_urban_extract.F').write_text('MODULE noahmp_urban_extract\nCONTAINS\n'+noah+'END MODULE\n')
        (build/'module_ra_gfdleta_extract.F').write_text('MODULE module_ra_gfdleta\nCONTAINS\n'+cal+'END MODULE\n')
        pflags=flags+record['kind_flags'][precision]
        files=[Path(s).name for s in base.SOURCES];files[2:2]=['_wrfstubs.f90']
        files+=['noahmp_tables_stub.f90','module_ra_gfdleta_extract.F','noahmp_urban_extract.F']
        for src in files:
            if a.relink:
                assert (build/src).with_suffix('.o').exists(),src
                continue
            base.guard();subprocess.run([str(fc),*pflags,'-c',src],cwd=build,check=True,timeout=1200)
        objects=[str(Path(s).with_suffix('.o')) for s in files]
        base.guard()
        subprocess.run([str(fc),*pflags,'-DWITH_BEM','-o','oracle3',str(HERE/'pvgr_oracle.F90'),*objects],cwd=build,check=True,timeout=1200)
        record[precision+'_oracle_sha256']=base.sha(build/'oracle3')
    (a.build/'build_record.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(record)[:300])


if __name__=='__main__':
    main()
