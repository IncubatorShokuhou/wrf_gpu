"""Stage the authentic Madrid case for a BEP+BEM PV/green-roof GPU smoke (lane dir only).

python proofs/w41_writergaps/stage_gpu_case.py ARM OUT_DIR
-> OUT_DIR/input (inputs symlinked read-only, private namelist: sf_urban_physics=3, YSU, revised-MM5,
   URBPARM.TBL = pristine + run_oracle.ARMS[ARM]) and OUT_DIR/wrf_root (GPUWRF_WRF_ROOT tree).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_oracle import ARMS, table  # noqa: E402

CASE = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual/urban_madrid/input')
PRISTINE = Path('<USER_HOME>/src/wrf_pristine/WRF')


def main():
    arm, out = sys.argv[1], Path(sys.argv[2])
    tbl = table(ARMS[arm])
    root = out/'wrf_root';(root/'run').mkdir(parents=True, exist_ok=True)
    for entry in PRISTINE.iterdir():
        if entry.name != 'run' and not (root/entry.name).exists():
            (root/entry.name).symlink_to(entry)
    for entry in (PRISTINE/'run').iterdir():
        if entry.name != 'URBPARM.TBL' and not (root/'run'/entry.name).exists():
            (root/'run'/entry.name).symlink_to(entry.resolve())
    (root/'run'/'URBPARM.TBL').write_text(tbl)
    case = out/'input';case.mkdir(parents=True, exist_ok=True)
    for entry in CASE.iterdir():
        if entry.name not in ('namelist.input', 'URBPARM.TBL') and not (case/entry.name).exists():
            (case/entry.name).symlink_to(entry.resolve())
    (case/'URBPARM.TBL').write_text(tbl)
    text = (CASE/'namelist.input').read_text()
    for key, value in {'sf_urban_physics': 3, 'bl_pbl_physics': 1, 'sf_sfclay_physics': 1}.items():
        text, n = re.subn(rf'^\s*{key}\s*=.*$', f' {key}={value},', text, flags=re.M | re.I)
        assert n == 1, key
    (case/'namelist.input').write_text(text)
    record = {'arm': arm, 'changes': ARMS[arm], 'case': str(CASE),
              'urbparm_sha256': hashlib.sha256(tbl.encode()).hexdigest(),
              'namelist_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'inputs': {n: hashlib.sha256((CASE/n).read_bytes()).hexdigest() for n in ('wrfinput_d01', 'wrfbdy_d01')}}
    (out/'stage.json').write_text(json.dumps(record, indent=2)+'\n');print(json.dumps(record))


if __name__ == '__main__':
    main()
