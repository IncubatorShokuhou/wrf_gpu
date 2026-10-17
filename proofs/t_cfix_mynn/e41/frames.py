import sys, re
from pathlib import Path
sys.path.insert(0, "src")
from gpuwrf.runtime.aot_executable import _proto_fields
proto = Path(sys.argv[1]).read_bytes()
sfi = [v for f, w, v in _proto_fields(proto) if f == 17][0]
files, funcs, locs, frames = [], [], [], []
for f, w, v in _proto_fields(sfi):
    if f == 1: files.append(v.decode())
    elif f == 2: funcs.append(v.decode())
    elif f == 3:
        d = {ff: vv for ff, ww, vv in _proto_fields(v)}; locs.append((d.get(1, 0), d.get(2, 0), d.get(3, 0)))
    elif f == 4:
        d = {ff: vv for ff, ww, vv in _proto_fields(v)}; frames.append((d.get(1, 0), d.get(2, 0)))
for a in sys.argv[2:]:
    fid = int(a); print("frame", fid)
    n = 0
    while fid and n < 14:
        loc_id, parent = frames[fid - 1]; fn, fu, line = locs[loc_id - 1]
        print("   ", re.sub(r".*/src/gpuwrf/", "", files[fn - 1]), line, funcs[fu - 1]); fid = parent; n += 1
