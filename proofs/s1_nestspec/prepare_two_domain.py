"""Stage a short, immutable WN3 two-domain CPU-WRF reference in the lane directory."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

from netCDF4 import Dataset, chartostring

SOURCE = Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
TABLES = Path('<USER_HOME>/src/wrf_pristine/WRF/run')
EXECUTABLE = Path('<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/bin/wrf')
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-nestspec/coupled')
mp = int(sys.argv[1])
run = ROOT / f'mp{mp}'
run.mkdir(parents=True, exist_ok=False)
binary = EXECUTABLE.read_bytes()
for forbidden in (b'WRFGPU2_DUMP', b'<DATA_ROOT>/wrf_gpu2/wrf_truth', b'em_quarter_ss_center_dump'):
    assert forbidden not in binary, forbidden
shutil.copy2(EXECUTABLE, run / 'wrf.exe')
files = {}
for name in ('wrfinput_d01', 'wrfinput_d02', 'wrfbdy_d01'):
    path = SOURCE / name
    shutil.copy2(path, run / name)
    files[name] = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
for path in TABLES.iterdir():
    if path.is_file() and path.name not in ('namelist.input', 'namelist.output', 'wrf.exe', 'real.exe', 'ndown.exe'):
        if not path.name.startswith(('wrfinput', 'wrfout', 'wrfbdy', 'rsl.', 'met_em')):
            (run / path.name).symlink_to(path.resolve())
with Dataset(run / 'wrfinput_d01') as ds:
    start = datetime.strptime(str(chartostring(ds['Times'][:])[0]), '%Y-%m-%d_%H:%M:%S')
end = start + timedelta(seconds=108)
text = (SOURCE / 'namelist.input').read_text()
edits = {'time_control': {'run_days':0,'run_hours':0,'run_minutes':0,'run_seconds':108,
          'input_from_file':'.true., .true.', 'history_interval':'0, 0', 'history_interval_s':'54, 18',
          'frames_per_outfile':'1, 1', 'restart':'.false.', 'restart_interval':10000},
         'domains': {'max_dom':2, 'nproc_x':1, 'nproc_y':1},
         'physics': {'mp_physics':f'{mp}, {mp}', 'cu_physics':'0, 0', 'sst_update':0}}
for prefix, date in (('start',start),('end',end)):
    for name, value in (('year',date.year),('month',date.month),('day',date.day),('hour',date.hour),
                        ('minute',date.minute),('second',date.second)):
        edits['time_control'][f'{prefix}_{name}']=f'{value}, {value}'
for group, settings in edits.items():
    for key, value in settings.items():
        line = f' {key} = {value},'
        pat = re.compile(rf'^\s*{key}\s*=.*$',re.M|re.I)
        if pat.search(text):
            text=pat.sub(line,text)
        else:
            text=re.sub(rf'(&{group}\s*\n)',lambda m:m.group(1)+line+'\n',text,count=1,flags=re.I)
(run / 'namelist.input').write_text(text)
plan = {'created_utc':datetime.now(timezone.utc).isoformat(), 'source_case':str(SOURCE),
        'case':str(run),'mp_physics':mp,'domains':2,'geometry':'120x70 +267x117,44levels,ratio3,start16/16',
        'parent_dt_s':54,'child_dt_s':18,'parent_steps':2,'child_steps':6,'duration_s':108,
        'reference':'original CPU-WRF v4.7.1 canonical RD11 executable, lane copy; no shared dump strings',
        'executable_sha256':hashlib.sha256(binary).hexdigest(),
        'input_files':files,'namelist_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'cache':'NONE','execution':'CPU24 nice19 MPI1 OMP1; no GPU',
        'gate':{'core':'unchanged frozen D6 manifest, separately boundary rings0..4/interior>=5',
                'extra_mass':{'atol':1.5e-5,'rtol':0,'basis':'frozen Gate2 mass-field absolute bound'},
                'extra_numbers':{'atol':1.,'rtol':5e-4,'basis':'predeclared Gate2 cloud-number bound, applied pointwise'},
                'extra_volumes':{'atol':1e-14,'rtol':5e-4,'basis':'predeclared volume absolute floor and same relative number-density bound'},
                'finite':'all raw arrays required finite; no clamps/masking/threshold revision'}}
(run / 'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'case':str(run),'start':str(start),'end':str(end),'executable_sha256':plan['executable_sha256']},indent=2))
