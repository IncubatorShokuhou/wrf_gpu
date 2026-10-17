"""Original WPS/real with sf_ocean=1; no editing incompatible WRF input headers."""
import json
import os
from pathlib import Path
from prepare import ROOT,PRISTINE,SLOTS,sha,write_json,private_executable
from extra_cases import stage
from gpuwrf.onboard.region import plan_region,namelists
from gpuwrf.onboard.forcing import retrieve_gfs
from gpuwrf.onboard.pipeline import _exe,_stage,_metgrid_levels,check_real
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset
import numpy as np
assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
work=ROOT/'atlantic_oml_real';work.mkdir(exist_ok=False)
wps=Path('<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/WPS')
geog=Path('<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/geog')
plan=plan_region([-17.1,27.7,-16.0,28.6],3.,'2026-10-13T00',24)
receipt=dict(plan=plan,status='PREPARING',source='public original GFS + unchanged NCAR geography/original WPS/real',
    reason='WN3 original sf_ocean0 input headers correctly refuse sf_ocean1; no header fabrication')
try:
    for name in ('geogrid','metgrid'):
        (work/name).mkdir()
        table=(wps/name/(name.upper()+'.TBL.ARW')).resolve()
        (work/name/(name.upper()+'.TBL')).symlink_to(table)
    (work/'Vtable').symlink_to((wps/'ungrib/Variable_Tables/Vtable.GFS').resolve())
    text,_=namelists(plan,geog);(work/'namelist.wps').write_text(text)
    for p in (PRISTINE/'run').iterdir():
        if p.is_file() and p.name not in ('namelist.input','wrf.exe','real.exe','ndown.exe','tc.exe'):(work/p.name).symlink_to(p.resolve())
    receipt['forcing']=retrieve_gfs(plan,work/'grib',300_000_000)
    for index,r in enumerate(receipt['forcing']):
        letters=''.join(chr(65+(index//26**p)%26) for p in (2,1,0))
        (work/('GRIBFILE.'+letters)).symlink_to(r['path'])
    for name in ('geogrid','ungrib','metgrid'):
        _stage([str(_exe(wps,name))],work,name,300)
    from prepare import render
    groups=parse_namelist_text=namelists(plan,geog,_metgrid_levels(work,plan))[1]
    (work/'namelist.input').write_text(groups)
    groups=parse_namelist(work/'namelist.input')
    groups['physics'].update(sf_ocean_physics=1,oml_hml0=20.,oml_gamma=.14)
    (work/'namelist.input').write_text(render(groups))
    _stage([str(_exe(PRISTINE,'real'))],work,'real',300)
    receipt['real_check']=check_real(work,plan)
    with Dataset(work/'wrfinput_d01') as ds:
        water=np.asarray(ds['XLAND'][0])>1.5
        assert water.any() and int(ds.SF_OCEAN_PHYSICS)==1
        receipt['water_census']=dict(cells=int(water.sum()),min_tsk=float(ds['TSK'][0][water].min()),max_tsk=float(ds['TSK'][0][water].max()))
    matrix=json.loads((ROOT/'MATRIX.json').read_text())
    for ident,flux,pbl,sf in (('c84',1,5,5),('c85',2,1,1)):
        g=parse_namelist(work/'namelist.input');g['physics'].update(isftcflx=flux,bl_pbl_physics=pbl,sf_sfclay_physics=sf)
        row=dict(id=ident,geography='authentic-Atlantic-original-OML-real',hours=24,domains=['d01'],
            options={k:g['physics'][k] for k in SLOTS},extra={},features=['OML1',f'hurricane-flux{flux}'],
            active_geography_census=receipt['water_census'])
        row=stage(row,work,g);private_executable(ROOT/ident,ROOT/'reference_zero')
        matrix['cases'].append(row);write_json(ROOT/ident/'inputs.json',row)
        old=next(r for r in matrix['cases'] if r['id']==('c58' if ident=='c84' else 'c59'))
        old.update(gpu_eligible=False,control_reason='Original sf_ocean input-header refusal; replaced by original-real sf_ocean1 '+ident)
        write_json(ROOT/old['id']/'inputs.json',old)
    write_json(ROOT/'MATRIX.json',matrix)
    receipt['input_sha256']={name:sha(work/name) for name in ('wrfinput_d01','wrfbdy_d01','namelist.input','geo_em.d01.nc')}
    receipt['status']='COMPLETE_ORIGINAL_OML_GEOGRAPHY'
finally:write_json(ROOT/'VP48-atlantic.json',receipt)
print(json.dumps(receipt,indent=2))
