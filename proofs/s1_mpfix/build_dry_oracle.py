"""Full pristine mp40 wrapper, dry/active-column GOTO200/GOTO400 coverage."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / 'data/wrf_pristine/WRF'
work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True, exist_ok=True)
save = ROOT / 'proofs/s1_mpfix/fixtures'
save.mkdir(exist_ok=True)
source = ROOT / 'proofs/v023/oracle/morraero'
driver = (source / 'morraero_oracle_driver.f90').read_text().replace('AERCU_OPT = 2', 'AERCU_OPT = 0')
# All existing driver/wrapper plumbing retained. Controlled inputs are inserted
# before its input savepoint, after build_aero (which otherwise seeds Nc).
marker = '  ! save inputs\n'
patch = '''  ! s1-mpfix: dry warm/cold, active warm/cold, mixed and saturated-empty.
  IF (case_id >= 8) THEN
    QC=0.; QR=0.; QI=0.; QS=0.; QG=0.
    NI=0.; NR=0.; NS=0.; NG=0.; NC=123456.
    P=80000.; PII=1.; TH=290.; QV=0.001; DZ=250.; W=0.
    IF (case_id == 9 .OR. case_id == 11) THEN
      TH=250.; QV=0.00001
    END IF
    IF (case_id == 10) QC(1,20,1)=0.001
    IF (case_id == 11) QI(1,20,1)=0.0001
    IF (case_id == 12) QV=0.03
    RHO=P/(287.*TH)
  END IF
'''
assert driver.count(marker) == 1
driver = driver.replace(marker, patch + marker)
receipt = {'cases': {'8':'warm empty dry', '9':'cold empty dry', '10':'one warm cloudy level amid dry levels',
                     '11':'one cold icy level amid dry levels', '12':'empty supersaturated warm'},
           'source_hashes': {}, 'builds': {}}
fc = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
for mode in ('fp32', 'fp64'):
    build = work / mode
    build.mkdir(exist_ok=True)
    files = [source/'morraero_stub_modules.f90', WRF/'share/module_model_constants.F',
             WRF/'phys/module_mp_radar.F', WRF/'phys/module_mp_morr_two_moment_aero.F']
    for path in files:
        shutil.copyfile(path, build/path.name)
        receipt['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (build/'driver.f90').write_text(driver)
    flags = ['-O2','-ffree-form','-ffree-line-length-none','-ffpe-summary=none','-cpp']
    if mode == 'fp64':
        flags += ['-fdefault-real-8','-fdefault-double-8']
    for name in [p.name for p in files] + ['driver.f90']:
        subprocess.run([fc, *flags, '-c', name], cwd=build, check=True)
    subprocess.run([fc,*flags,'-o','oracle',*[p.stem+'.o' for p in files],'driver.o'], cwd=build, check=True)
    for case in range(8,13):
        raw = build/f'case{case}.txt'
        with raw.open('w') as output:
            subprocess.run([str(build/'oracle'),str(case)], stdout=output,cwd=build,check=True)
        subprocess.run([sys.executable, str(source/'dump_to_json.py'), str(raw),str(save/f'dry_{mode}_{case}.json')],check=True)
    receipt['builds'][mode] = {'flags':flags,'driver_sha256':hashlib.sha256(driver.encode()).hexdigest()}
receipt['compiler'] = subprocess.check_output([fc,'--version'],text=True).splitlines()[0]
(save/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
