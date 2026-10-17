"""Real WN3 Atlantic root domains for OML/SST/hurricane/slope/GWD options."""
import json
import os
from pathlib import Path
import shutil
from prepare import ROOT, SLOTS, private_executable, sha, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset
import numpy as np

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix=json.loads((ROOT/'MATRIX.json').read_text())
base=Path('<USER_HOME>/wrf_gpu2_lanes/u2-twoway/cpu-reference')
flow=Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run/wrflowinp_d01')
owned=ROOT/'authentic_wrflowinp_d01'
if not owned.exists():shutil.copyfile(flow,owned)
assert sha(owned)==sha(flow)
with Dataset(base/'wrfinput_d01') as ds:
    water=ds['XLAND'][0]>1.5
    assert water.any()
    census=dict(water_cells=int(water.sum()),min_water_tsk=float(ds['TSK'][0][water].min()),
                max_water_tsk=float(ds['TSK'][0][water].max()))
for ident,flux,pbl,sf in (('c58',1,5,5),('c59',2,1,1),('c60',0,5,5)):
    if any(r['id']==ident for r in matrix['cases']):continue
    groups=parse_namelist(base/'namelist.input')
    for group in groups.values():
        for key,value in group.items():
            if isinstance(value,list):group[key]=value[0]
    groups['domains'].update(max_dom=1,feedback=0)
    groups['time_control'].update(run_days=0,run_hours=24,run_minutes=0,run_seconds=0,
        end_year=2026,end_month=3,end_day=1,end_hour=0,end_minute=0,end_second=0,
        history_interval=60,history_interval_s=0,auxinput4_interval=360,io_form_auxinput4=2)
    groups['physics'].update(mp_physics=8,cu_physics=0,bl_pbl_physics=pbl,sf_sfclay_physics=sf,
        sf_ocean_physics=0 if ident=='c60' else 1,isftcflx=flux,oml_hml0=20.,oml_gamma=.14,
        oml_relaxation_time=3600.,sst_update=1 if ident=='c60' else 0,slope_rad=1,topo_shading=1)
    groups['dynamics']['gwd_opt']=1
    row=dict(id=ident,geography='authentic-WN3-Atlantic-root',hours=24,domains=['d01'],
        options={k:groups['physics'][k] for k in SLOTS},extra={},
        features=(['sst_update1'] if ident=='c60' else ['OML1',f'hurricane-flux{flux}'])+['slope1+topo1','GWDO1'],
        active_geography_census=census)
    row=stage(row,base,groups)
    private_executable(ROOT/ident,ROOT/'reference_zero')
    (ROOT/ident/'input/wrflowinp_d01').symlink_to(owned)
    (ROOT/ident/'cpu/wrflowinp_d01').symlink_to(owned)
    row['input_sha256']={p.name:sha(p) for p in (ROOT/ident/'input').iterdir() if p.is_file()}
    row['reference_semantics']='compiler-defined zero, inactive canonical Morrison-aero object'
    write_json(ROOT/ident/'inputs.json',row)
    matrix['cases'].append(row);write_json(ROOT/'MATRIX.json',matrix)
    print(json.dumps({'case':ident,'errors':row['admission_errors'],'census':census}),flush=True)
