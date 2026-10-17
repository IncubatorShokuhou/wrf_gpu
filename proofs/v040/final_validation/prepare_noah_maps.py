"""Bounded original NCAR geography -> Madrid WPS/real; never edits map values."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
from prepare import ROOT, PRISTINE, sha, write_json
from gpuwrf.onboard.pipeline import _exe, _metgrid_levels, check_real
from gpuwrf.onboard.region import namelists
from gpuwrf.io.netcdf_lock import Dataset
import numpy as np

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
scratch=Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep')
deadline=datetime.fromisoformat('2026-10-14T11:28:46+00:00').timestamp()
download=json.loads((ROOT/'VP26-NCAR-download.json').read_text())
assert download['total_bytes']<=5_000_000_000
assert shutil.disk_usage(scratch).free>=120*2**30
geog=scratch/'geog';geog.mkdir(exist_ok=True)
receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=download['deadline_utc'],
    downloads=download,status='PREPARING',map_values='unchanged official NCAR geographic samples')
prior=ROOT/'VP27-NCAR-Madrid-maps.json'
if prior.exists():
    old=json.loads(prior.read_text())
    assert old['status']!='COMPLETE_ORIGINAL_MADRID_MAPS','Never overwrite completed original input'
    assert old['downloads']==download
    receipt=old
    receipt.setdefault('resumes',[]).append(datetime.now(timezone.utc).isoformat())


def check():
    while Path('/tmp/wrf_gpu2_quiet').exists():
        assert time.time()<deadline,'One-hour map time box exhausted during QUIET'
        time.sleep(.5)
    assert time.time()<deadline,'One-hour map time box exhausted'


def run(name,exe,work):
    check();start=time.monotonic()
    with (work/(name+'.log')).open('w') as log:
        p=subprocess.Popen([str(exe)],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        paused=False
        try:
            while p.poll() is None:
                quiet=Path('/tmp/wrf_gpu2_quiet').exists()
                if quiet!=paused:
                    os.killpg(p.pid,signal.SIGSTOP if quiet else signal.SIGCONT);paused=quiet
                assert time.time()<deadline and time.monotonic()-start<600,'Bounded WPS stage timed out'
                time.sleep(.2)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGCONT);os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        assert p.returncode==0,(name,p.returncode)
    receipt.setdefault('stages',[]).append(dict(name=name,rc=p.returncode,wall_s=time.monotonic()-start,
        executable=str(exe),executable_sha256=sha(exe)))


try:
    for record in download['files']:
        if any(r['name']==record['name'] for r in receipt.get('archives',[])):
            continue  # completed lossless extraction, same archive SHA receipt
        archive=Path(record['path']);assert sha(archive)==record['sha256']
        count,total=0,0
        with tarfile.open(archive,'r:*') as packed:
            for member in packed:
                check();relative=Path(member.name)
                assert not relative.is_absolute() and '..' not in relative.parts
                target=geog/relative
                if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                assert member.isfile(),member.name
                target.parent.mkdir(parents=True,exist_ok=True)
                with packed.extractfile(member) as incoming,target.open('wb') as out:
                    while True:
                        check();block=incoming.read(1<<20)
                        if not block:break
                        out.write(block)
                assert target.stat().st_size==member.size
                count+=1;total+=member.size
        receipt.setdefault('archives',[]).append(dict(name=record['name'],files=count,raw_bytes=total,
            compressed_sha256=record['sha256']))
        archive.unlink()  # Exact owned download; extracted maps are the dependent inputs.
        print(json.dumps(receipt['archives'][-1]),flush=True)
    mandatory=Path('<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/geog')
    for p in mandatory.iterdir():
        if p.is_dir() and not (geog/p.name).exists():(geog/p.name).symlink_to(p)
    source=Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual/urban_madrid/input')
    plan=json.loads((source/'onboarding.json').read_text())['plan']
    work=ROOT/'madrid_maps_real';work.mkdir(exist_ok=True)
    wps=Path('<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WPS')
    binaries=Path('<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/WPS')
    for name in ('geogrid','metgrid'):
        (work/name).mkdir(exist_ok=True)
        table=wps/name/(name.upper()+('.TBL.ARW.noahmp' if name=='geogrid' else '.TBL.ARW'))
        link=work/name/(name.upper()+'.TBL')
        if not link.exists():link.symlink_to(table)
        receipt.setdefault('table_sha256',{})[name]=sha(table)
    for p in source.glob('FILE:*'):
        if not (work/p.name).exists():(work/p.name).symlink_to(p.resolve())
    receipt['reused_original_ungrib_sha256']={p.name:sha(p) for p in work.glob('FILE:*')}
    assert len(receipt['reused_original_ungrib_sha256'])==9
    for p in (PRISTINE/'run').iterdir():
        if p.is_file() and p.name not in ('namelist.input','wrf.exe','real.exe','ndown.exe','tc.exe'):
            if not (work/p.name).exists():(work/p.name).symlink_to(p.resolve())
    wps_text,wrf_text=namelists(plan,geog)
    (work/'namelist.wps').write_text(wps_text)
    for name in ('geogrid','metgrid'):run(name,_exe(binaries,name),work)
    levels=_metgrid_levels(work,plan)
    (work/'namelist.input').write_text(namelists(plan,geog,levels)[1])
    run('real',_exe(PRISTINE,'real'),work)
    receipt['real_check']=check_real(work,plan)
    with Dataset(work/'wrfinput_d01') as ds:
        keys=('SOILCL1','SOILCL2','SOILCL3','SOILCL4','SOILCOMP','CROPTYPE','PLANTING','HARVEST','SEASON_GDD',
              'TD_FRACTION','IRFRACT','SIFRACT','MIFRACT','FIFRACT','IRRSPLH')
        receipt['map_census']={}
        for key in keys:
            a=np.asarray(ds[key][:])
            receipt['map_census'][key]=dict(nonzero=int(np.count_nonzero(a)),finite=bool(np.isfinite(a).all()),
                min=float(a.min()),max=float(a.max()))
        assert all(v['finite'] for v in receipt['map_census'].values())
        assert all(receipt['map_census'][k]['nonzero'] for k in keys[:5]),'Missing original soil maps'
    receipt['input_sha256']={p.name:sha(p) for p in work.iterdir() if p.name in ('wrfinput_d01','wrfbdy_d01','namelist.input','geo_em.d01.nc')}
    receipt['all_requested_maps_active']=all(receipt['map_census'][k]['nonzero'] for k in keys[:10])
    receipt['status']=('COMPLETE_ORIGINAL_MADRID_MAPS' if receipt['all_requested_maps_active'] else
                       'COMPLETE_WPS_REAL_PARTIAL_MAP_COVERAGE')
finally:
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(ROOT/'VP27-NCAR-Madrid-maps.json',receipt)
print(json.dumps(receipt,indent=2))
