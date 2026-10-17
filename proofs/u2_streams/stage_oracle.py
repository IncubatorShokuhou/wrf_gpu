"""Private original CPU-WRF 4-step auxhist7 oracle; no reference writes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
source = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
repo = Path(__file__).resolve().parents[2]
root.mkdir(parents=True,exist_ok=False)
for entry in source.iterdir():
    if entry.name.startswith(('wrfout','rsl.','namelist.','wrf.exe','tslist','auxhist','iofields')):
        continue
    if entry.is_file():
        (root/entry.name).symlink_to(entry.resolve())
tables = Path('<USER_HOME>/src/wrf_pristine/WRF/test/em_real/oracle_run_v090')
for entry in [*tables.glob('qr_acr*.dat'),tables/'freezeH2O.dat']:
    if not (root/entry.name).exists():
        (root/entry.name).symlink_to(entry.resolve())
exe = root/'wrf.exe'
shutil.copy2((source/'wrf.exe').resolve(),exe)
assert b'<DATA_ROOT>/wrf_gpu2/wrf_truth/' not in exe.read_bytes()
manifest = json.loads((repo/'proofs/u2_tslist/oracle/manifest.json').read_text())
text = manifest['namelist'].replace('&time_control\n','&time_control\n'
    ' auxhist7_interval = 0,\n auxhist7_interval_s = 18,\n frames_per_auxhist7 = 1,\n'
    ' io_form_auxhist7 = 2,\n auxhist7_outname = "auxhist7_d<domain>_<date>",\n'
    ' iofields_filename = "stream_fields.txt",\n ignore_iofields_warning = .false.,\n')
(root/'namelist.input').write_text(text)
fields = ['T2','Q2','U10','V10','PSFC','LH','HFX','PBLH','TSK','RAINC','RAINNC','SWDOWN','GLW']
(root/'stream_fields.txt').write_text('+:h:7:'+','.join(fields)+'\n')
(root/'STAGING.json').write_text(json.dumps({'reference':'original CPU-WRF auxhist7',
    'source':str(source),'executable_source':str((source/'wrf.exe').resolve()),
    'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
    'namelist':text,'iofields':(root/'stream_fields.txt').read_text(),'fields':fields,
    'work':'Swiss42x42x44 dt18 4steps; auxhist7 every18s; stations exact i20j20/i9j12; noGPU',
    'launch':'mpirun -np 1 --bind-to none ./wrf.exe',
    'tolerances':{'selection':'byte-exact same-input gather','window_mean_rtol':2e-7,'window_other':'byte-exact'}},indent=2)+'\n')
