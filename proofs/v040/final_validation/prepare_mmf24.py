"""Extend the authentic MMF lowland forcing using retained geogrid output."""
import json
import os
from pathlib import Path
import shutil
from prepare import ROOT, PRISTINE, sha, write_json
from gpuwrf.onboard.forcing import retrieve_gfs
from gpuwrf.onboard.pipeline import _exe, _stage, _metgrid_levels, check_real
from gpuwrf.onboard.region import namelists
from gpuwrf.io.netcdf_lock import Dataset
import numpy as np

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
original = Path('<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/MMF34_lowland')
wps = Path('<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/WPS')
work = ROOT / 'mmf24_real'
work.mkdir(exist_ok=False)
manifest = json.loads((original / 'onboarding.json').read_text())
plan = dict(manifest['plan'])
plan.update(hours=24, end='2026-10-13_00:00:00', boundary_hours=list(range(0,25,3)))
geo = original / 'geo_em.d01.nc'
receipt = dict(schema='v040-mmf24-original-preprocessing-v1', plan=plan,
    original_geography=str(geo), geography_sha256=sha(geo),
    original_geography_provenance=str(original / 'onboarding.json'),
    source='public original GFS20261012T00 + unchanged original NCAR WPS geo_em',
    status='PREPARING', executables={k:str(_exe(wps if k!='real' else PRISTINE,k))
                                    for k in ('ungrib','metgrid','real')})
try:
    # Geogrid does not run again; its exact retained result already contains
    # authentic groundwater geography. No replacement geographic values.
    (work / 'geo_em.d01.nc').symlink_to(geo)
    (work / 'metgrid').mkdir()
    (work / 'metgrid/METGRID.TBL').symlink_to((wps / 'metgrid/METGRID.TBL.ARW').resolve())
    (work / 'Vtable').symlink_to((wps / 'ungrib/Variable_Tables/Vtable.GFS').resolve())
    wps_text, wrf_text = namelists(plan, original)
    (work / 'namelist.wps').write_text(wps_text)
    (work / 'namelist.input').write_text(wrf_text)
    for p in (PRISTINE / 'run').iterdir():
        if p.is_file() and p.name not in ('namelist.input','wrf.exe','real.exe','ndown.exe','tc.exe'):
            (work / p.name).symlink_to(p.resolve())
    receipt['forcing'] = retrieve_gfs(plan, work / 'grib', 300_000_000)
    for index, row in enumerate(receipt['forcing']):
        letters = ''.join(chr(65+(index//26**power)%26) for power in (2,1,0))
        (work / f'GRIBFILE.{letters}').symlink_to(Path(row['path']))
    for name in ('ungrib','metgrid'):
        _stage([receipt['executables'][name]], work, name, 300)
    levels = _metgrid_levels(work, plan)
    (work / 'namelist.input').write_text(namelists(plan, original, levels)[1])
    _stage([receipt['executables']['real']], work, 'real', 300)
    receipt['real_check'] = check_real(work, plan)
    with Dataset(work / 'wrfinput_d01') as ds:
        ice = (ds['IVGTYP'][0] == int(ds.ISICE)) & (ds['XLAND'][0] < 1.5)
        assert not ice.any()
        receipt['groundwater_census'] = {n:dict(nonzero=int(np.count_nonzero(ds[n][:])),
            finite=bool(np.isfinite(ds[n][:]).all())) for n in ('FDEPTH','EQZWT','RECHCLIM','RIVERBED')}
        assert all(r['nonzero'] and r['finite'] for r in receipt['groundwater_census'].values())
    receipt['input_sha256'] = {n:sha(work/n) for n in ('wrfinput_d01','wrfbdy_d01','namelist.input')}
    receipt['status'] = 'COMPLETE_CPU_PREPROCESSING'
finally:
    write_json(work / 'receipt.json', receipt)
print(json.dumps({'status':receipt['status'],'download_bytes':sum(r['bytes'] for r in receipt['forcing']),
                  'groundwater_census':receipt.get('groundwater_census')}))
