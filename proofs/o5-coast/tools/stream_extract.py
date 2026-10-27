"""Streaming per-frame extraction + guarded deletion of PUBLISHED wrfout files during a GPU arm (o5-coast).

usage: python stream_extract.py WRFOUT_DIR FRAME_DIR --pid PID [--no-delete] [--poll 5]
The gpuwrf writer publishes each history file by an atomic rename of a hidden '.<name>...tmp' file, so a file with
its final name 'wrfout_d0N_YYYY-MM-DD_HH:MM:SS' is complete. For each such file: extract load_raw.frame() fields,
require all finite, write FRAME_DIR/<name>.npz atomically, append {name, bytes, sha256, mtime} to
FRAME_DIR/MANIFEST.jsonl, then (unless --no-delete) unlink exactly that file. Deletion guards: realpath of the file
must be a direct child of realpath(WRFOUT_DIR), WRFOUT_DIR must lie under <DATA_ROOT>/wrf_gpu2/scratch/o5-coast/,
name must match the wrfout pattern, file must not be a symlink. Exits after PID ended and no published file remains.
Keeps the arm's disk peak to a few frames instead of ~20 GiB (manager DISK rule 2026-10-24). No JAX."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_raw import frame  # noqa: E402

PAT = re.compile(r'^wrfout_d0[1-9]_\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d$')
ROOT = '<DATA_ROOT>/wrf_gpu2/scratch/o5-coast/'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wrfout_dir'); ap.add_argument('frame_dir')
    ap.add_argument('--pid', type=int, required=True); ap.add_argument('--no-delete', action='store_true')
    ap.add_argument('--poll', type=float, default=5.0)
    a = ap.parse_args()
    wdir = Path(os.path.realpath(a.wrfout_dir)); fdir = Path(a.frame_dir); fdir.mkdir(parents=True, exist_ok=True)
    delete = not a.no_delete
    if delete and not str(wdir).startswith(ROOT):
        sys.exit(f'refusing deletion outside {ROOT}: {wdir}')
    man = fdir / 'MANIFEST.jsonl'
    done = set()
    if man.exists():
        done = {json.loads(line)['name'] for line in man.read_text().splitlines() if line.strip()}
    n = 0
    while True:
        names = sorted(p.name for p in wdir.iterdir() if PAT.match(p.name)) if wdir.exists() else []
        todo = [x for x in names if x not in done]
        for name in todo:
            p = wdir / name
            if p.is_symlink() or Path(os.path.realpath(p)).parent != wdir:
                sys.exit(f'refusing non-regular/foreign file {p}')
            st = p.stat()
            digest = sha256(p)
            fr = frame(p)
            bad = [k for k, v in fr.items() if k != 'times' and np.issubdtype(np.asarray(v).dtype, np.floating)
                   and not np.isfinite(v).all()]
            tmp = fdir / f'.{name}.tmp.npz'
            np.savez_compressed(tmp, **fr)
            os.replace(tmp, fdir / f'{name}.npz')
            rec = dict(name=name, bytes=st.st_size, sha256=digest, mtime=st.st_mtime, nonfinite_fields=bad,
                       deleted=False, extracted_utc=time.time())
            if delete and sha256(p) == digest:
                os.unlink(p)
                rec['deleted'] = True
            with open(man, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            done.add(name); n += 1
        if not todo and not alive(a.pid):
            names = sorted(p.name for p in wdir.iterdir() if PAT.match(p.name)) if wdir.exists() else []
            if all(x in done for x in names):
                break
        time.sleep(a.poll)
    print(json.dumps(dict(extracted=n, manifest=str(man))))


if __name__ == '__main__':
    main()
