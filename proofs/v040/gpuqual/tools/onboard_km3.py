"""Use merged onboarding and original real.exe for genuine RUC/urban GFS cases."""
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from gpuwrf.onboard import pipeline
from gpuwrf.onboard.region import plan_region

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert os.sched_getaffinity(0) == {29}
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual')
original = Path('<USER_HOME>/src/wrf_pristine/WRF')
private = ROOT / 'pristine'
real = private / 'real.exe'
if not real.exists():
    shutil.copyfile(original / 'main/real.exe', real)
    real.chmod(0o755)
if not (private / 'run').exists():
    (private / 'run').symlink_to(original / 'run')
wps = ROOT / 'WPS'
wps.mkdir(exist_ok=True)
upstream = Path('<USER_HOME>/wrf_gpu2_lanes/u1-onboard/WPS')
for item in upstream.iterdir():
    target = wps / item.name
    if not target.exists():
        target.symlink_to(item.resolve())
geog = Path('<DATA_ROOT>/canairy_meteo/artifacts/wps_geog/WPS_GEOG_LOW_RES')
plan = plan_region([-4.65, 39.69, -2.75, 41.14], 9., '2026-10-13T00', 24, 12)
plan['source'] = 'gfs'
base_namelists = pipeline.namelists
selection = {}
def selected_namelists(*args, **kwargs):
    a, b = base_namelists(*args, **kwargs)
    for key, value in selection.items():
        b, count = re.subn(r'(?im)(\b' + key + r'\s*=\s*)[^,\n/]+', lambda m: m[1] + str(value), b)
        if count == 0:
            if key != 'sf_urban_physics':
                raise ValueError(f'No unique onboarding field {key}: {count}')
            b = b.replace('&physics', '&physics\n sf_urban_physics=1,', 1)
        elif count != 1:
            raise ValueError(f'Ambiguous onboarding field {key}: {count}')
    return a, b
pipeline.namelists = selected_namelists
stage = pipeline._stage
def quiet_stage(argv, *args, **kwargs):
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(1)
    return stage(['bash', '-c', 'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"', 'onboard-stage'] + argv, *args, **kwargs)
pipeline._stage = quiet_stage
receipts = json.loads((ROOT / 'CITY_ONBOARDING.json').read_text())
for name, selection in (
        ('km3_madrid', {'diff_opt': 2, 'km_opt': 3}),):
    case = ROOT / name
    case.mkdir(exist_ok=True)
    args = SimpleNamespace(case_dir=case / 'input', geog_dir=geog, wps_root=wps,
                           wrf_root=private, source='gfs', max_download_mb=200, stage_timeout=1800)
    manifest = pipeline.prepare(plan, args)
    manifest['qualification_overrides'] = selection
    manifest['geography_reused_read_only'] = str(geog)
    manifest['original_real_sha256'] = hashlib.sha256((original / 'main/real.exe').read_bytes()).hexdigest()
    manifest['private_real_sha256'] = hashlib.sha256(real.read_bytes()).hexdigest()
    assert manifest['private_real_sha256'] == manifest['original_real_sha256']
    (case / 'input/onboarding.json').write_text(json.dumps(manifest, indent=2) + '\n')
    transferred = sum(row['bytes'] for row in manifest['forcing'])
    receipts.append({'config': name, 'plan': plan, 'overrides': selection,
                     'forcing_download_bytes': transferred,
                     'manifest': str(case / 'input/onboarding.json')})
    assert sum(row['forcing_download_bytes'] for row in receipts) < 2_000_000_000
    (ROOT / 'CITY_ONBOARDING.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(name, manifest['status'], plan['nx'], plan['ny'], transferred, flush=True)
