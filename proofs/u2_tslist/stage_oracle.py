"""Stage a private short Swiss CPU-WRF run; never execute in a reference directory."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

root = Path(sys.argv[1]).resolve()
source = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
root.mkdir(parents=True, exist_ok=False)
for entry in source.iterdir():
    if entry.name.startswith(('wrfout', 'rsl.', 'namelist.', 'wrf.exe')):
        continue
    if entry.is_file():
        (root / entry.name).symlink_to(entry.resolve())
for entry in Path('<USER_HOME>/src/wrf_pristine/WRF/test/em_real/oracle_run_v090').glob('qr_acr*.dat'):
    if not (root / entry.name).exists():
        (root / entry.name).symlink_to(entry.resolve())
(root / 'freezeH2O.dat').symlink_to('<USER_HOME>/src/wrf_pristine/WRF/test/em_real/oracle_run_v090/freezeH2O.dat')
exe_source = (source / 'wrf.exe').resolve()
exe = root / 'wrf.exe'
shutil.copy2(exe_source, exe)
# E104: refuse an executable carrying the instrumented pristine build's shared dump paths.
assert b'<DATA_ROOT>/wrf_gpu2/wrf_truth/' not in exe.read_bytes()
text = (source / 'namelist.input').read_text()
text = text.replace('run_days                            = 1', 'run_days                            = 0')
text = text.replace('run_seconds                         = 0', 'run_seconds                         = 72')
text = text.replace('end_day                             = 16', 'end_day                             = 15')
text = text.replace('end_minute                          = 0', 'end_minute                          = 1')
text = text.replace('end_second                          = 0', 'end_second                          = 12')
text = text.replace('&time_control\n', '&time_control\n history_interval_s = 18,\n')
text = text.replace('history_interval                    = 60', 'history_interval                    = 0')
text = text.replace('&domains\n', '&domains\n ts_buf_size = 3,\n max_ts_locs = 5,\n max_ts_level = 5,\n')
ij_unstagger = len(sys.argv) > 2 and sys.argv[2] == '--ij-unstagger'
if ij_unstagger:
    text = text.replace('&domains\n', '&domains\n tslist_unstagger_winds = .true.,\n')
(root / 'namelist.input').write_text(text)
# Coordinates chosen from the Swiss grid, rounded as required by WRF's F7.3/F8.3 input.
from netCDF4 import Dataset
with Dataset(source / 'wrfinput_d01') as ds:
    lat, lon = ds['XLAT'][0], ds['XLONG'][0]
    stations = [('Swiss centre', 'ctr', 20, 20), ('Swiss west', 'west', 9, 12)]
    lines = ['#-----------------------------------------------#',
             '# 24 characters for name | pfx |  LAT  |   LON  |',
             '#-----------------------------------------------#']
    if ij_unstagger:
        lines[1] = '# 24 characters for name | pfx |   I   |    J   |'
        for desc, prefix, i, j in stations:
            lines.append(f'{desc:25s} {prefix:5s} {i:7d} {j:8d}')
        lines.append(f'{"Outside":25s} {"out":5s} {999:7d} {999:8d}')
    else:
        for desc, prefix, i, j in stations:
            lines.append(f'{desc:25s} {prefix:5s} {lat[j-1,i-1]:7.3f} {lon[j-1,i-1]:8.3f}')
        lines.append(f'{"Outside":25s} {"out":5s} {0:7.3f} {0:8.3f}')
    (root / 'tslist').write_text('\n'.join(lines) + '\n')
receipt = {'reference': str(source), 'executable_source': str(exe_source),
           'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
           'namelist': text, 'tslist': (root / 'tslist').read_text(),
           'shared_instrumented_dump_paths': False, 'work': 'Swiss42x42x44 dt18 4steps ts_buf_size3 level5'}
(root / 'STAGING.json').write_text(json.dumps(receipt, indent=2) + '\n')
