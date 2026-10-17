"""Stage lane-owned WRF ingestion probes; never modify shared references.

Usage: prepare_reference.py LANE_ROOT
TF02 has matched 3-step old/repaired metadata arms with 18s restart savepoints.
TF03 uses the repaired file for a full 24h reference to the completed GPU arm.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import shutil
import sys

from gpuwrf.io.netcdf_lock import Dataset


def edit(text, group, values):
    match = re.search(rf"&{group}\b(.*?)/", text, re.S | re.I)
    body = match.group(1)
    for key, value in values.items():
        pattern = rf"^\s*{key}\s*=.*$"
        line = f" {key} = {value},"
        body = re.sub(pattern, line, body, flags=re.M | re.I) if re.search(pattern, body, re.M | re.I) else body + line + "\n"
    return text[:match.start()] + f"&{group}{body}/" + text[match.end():]


def main():
    root = Path(sys.argv[1]).resolve()
    ref = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual/fdda1_fixed')
    root.mkdir(parents=True, exist_ok=True)
    private = root/'p'
    private.mkdir(exist_ok=True)
    original = Path('<USER_HOME>/src/wrf_pristine/WRF/main/wrf.exe')
    raw = original.read_bytes()
    patched = raw
    patches = []
    for i, old in enumerate((b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt',
                             b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt')):
        new = str(private/str(i)).encode().ljust(len(old), b' ')
        assert len(new) == len(old) and patched.count(old) == 1
        patches.append({'old':old.decode(), 'new':new.decode(), 'offset':patched.index(old)})
        patched = patched.replace(old, new)
    exe = private/'wrf.exe'
    exe.write_bytes(patched)
    exe.chmod(0o755)
    tables = root/'tables'
    tables.mkdir(exist_ok=True)
    # These files can be written by Thompson: copy to the lane, never modify references.
    table_names = ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat')
    for n in table_names:
        if not (tables/n).exists():
            shutil.copyfile(ref/'cpu'/n, tables/n)
    analysis = root/'wrffdda_d01'
    shutil.copyfile(ref/'input/wrffdda_d01', analysis)
    spec = importlib.util.spec_from_file_location('make_wrffdda', Path(__file__).resolve().parents[1]/'o2_fdda/make_wrffdda.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with Dataset(analysis, 'a') as ds:
        helper.wrf_metadata(ds, ref/'cpu/wrfout_d01_2023-01-15_00:00:00', str(ref/'input/wrffdda_d01'))
    variables = {}
    with Dataset(ref/'input/wrffdda_d01') as old, Dataset(analysis) as new:
        for n in old.variables:
            a, b = old[n][:], new[n][:]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), n
            variables[n] = {'sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                            'old_attrs':{k:str(old[n].getncattr(k)) for k in old[n].ncattrs()},
                            'new_attrs':{k:str(new[n].getncattr(k)) for k in new[n].ncattrs()}}
    for arm in ('old','repaired','full24'):
        case = root/arm
        case.mkdir()
        for p in (ref/'cpu').iterdir():
            if p.name.startswith(('wrfout','wrfrst','rsl.','namelist','wrf.exe','wrffdda')) or p.name in table_names:
                continue
            if p.is_file():
                (case/p.name).symlink_to(p.resolve())
        for n in table_names:
            (case/n).symlink_to(tables/n)
        (case/'wrf.exe').symlink_to(exe)
        (case/'wrffdda_d01').symlink_to((ref/'input/wrffdda_d01').resolve() if arm == 'old' else analysis)
        text = (ref/'input/namelist.input').read_text()
        if arm != 'full24':
            text = edit(text,'time_control',dict(run_days=0,run_hours=0,run_minutes=0,run_seconds=54,
                end_day=15,end_hour=0,end_minute=0,end_second=54,history_interval=0,history_interval_s=18,
                history_interval_m=0,restart_interval=0,restart_interval_s=18,restart_interval_m=0))
        text = edit(text,'time_control',dict(debug_level=10 if arm != 'full24' else 0))
        (case/'namelist.input').write_text(text)
    receipt = dict(original=str(original),original_sha256=hashlib.sha256(raw).hexdigest(),
        private_sha256=hashlib.sha256(patched).hexdigest(),patches=patches,
        arithmetic_unchanged=True,all13_variable_payloads_byte_equal=True,variables=variables,
        source_ref=str(ref),core=1,dt_s=18,short_steps=3,full_hours=24,cache='NONE')
    (root/'PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='variables'},indent=2))


if __name__ == '__main__':
    main()
