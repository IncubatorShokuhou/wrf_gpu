"""Link an admitted v-prep composite into the lane case root (no copies, no JAX): <root>/<case>/input/*
-> v-prep inputs, <root>/<case>/cpu/wrfout_* -> the existing ORIGINAL CPU-WRF reference frames.
usage: python link_vprep_case.py <cases_root> <case> [...]"""
import hashlib, json, sys
from pathlib import Path

VPREP = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
root = Path(sys.argv[1])
for case in sys.argv[2:]:
    src, dst = VPREP / case, root / case
    assert not (dst / 'gpu').exists(), f'{case}: gpu dir exists (E42)'
    for sub in ('input', 'cpu'):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    rec = dict(case=case, source=str(src), inputs={}, cpu_frames=0)
    for f in sorted((src / 'input').iterdir()):
        link = dst / 'input' / f.name
        if not link.is_symlink():
            link.symlink_to(f.resolve())
        rec['inputs'][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    for f in sorted((src / 'cpu').glob('wrfout_d0*')):
        link = dst / 'cpu' / f.name
        if not link.is_symlink():
            link.symlink_to(f.resolve())
        rec['cpu_frames'] += 1
    (dst / 'LINK.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(case, rec['cpu_frames'], 'frames', list(rec['inputs']))
