"""Record-level comparison of v0.3.3 compiled-thunk (Riegeli) fused blobs: decompress every record, diff pairwise,
and test whether all differing bytes of each record lie inside the TMN payload(s) located in that record.
usage: JAX_PLATFORMS=cpu python record_compare.py SNAP OUT_JSON L0227.npz L0707.npz LABEL=BLOB ...
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

snap, out_json, l27, l07 = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
blobs = dict(a.split("=", 1) for a in sys.argv[5:])
sys.path.insert(0, f"{snap}/src")
sys.path.insert(0, f"{snap}/scripts")
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm  # noqa: E402
import wave_cubin_gate as wcg  # noqa: E402

leaves = {"0227": np.load(l27), "0707": np.load(l07)}
PAYLOADS = {}
for case, lv in leaves.items():
    for k in ("d02|noahmp_case.tbot", "d03|noahmp_case.tbot"):
        for dt in (np.float32, np.float64):
            PAYLOADS[f"{case}:{k}:{np.dtype(dt).name}"] = np.ascontiguousarray(lv[k].astype(dt)).tobytes()


def records(b):
    out = []
    i = 0
    while True:
        try:
            out.append(wcg._record(b, i, ax, asm))
        except (ValueError, StopIteration):
            break
        i += 1
    return out


def find_all(hay, pay):
    pos, start = [], 0
    while (j := hay.find(pay, start)) >= 0:
        pos.append(j)
        start = j + 1
    return pos


recs = {}
res = {"blobs": {}, "pairs": {}}
for label, path in blobs.items():
    rs = records(Path(path).read_bytes())
    recs[label] = rs
    res["blobs"][label] = {"n_records": len(rs), "record_len": [len(r) for r in rs],
                           "record_sha16": [hashlib.sha256(r).hexdigest()[:16] for r in rs],
                           "tmn_hits": [{name: find_all(r, p) for name, p in PAYLOADS.items() if find_all(r, p)} for r in rs]}
labels = list(blobs)
for i, a in enumerate(labels):
    for b in labels[i + 1:]:
        pr = []
        for k, (ra, rb) in enumerate(zip(recs[a], recs[b])):
            row = {"record": k, "identical": ra == rb, "len_a": len(ra), "len_b": len(rb)}
            if ra != rb and len(ra) == len(rb):
                x = np.frombuffer(ra, np.uint8) != np.frombuffer(rb, np.uint8)
                idx = np.flatnonzero(x)
                spans = []
                for name, offs in list(res["blobs"][a]["tmn_hits"][k].items()) + list(res["blobs"][b]["tmn_hits"][k].items()):
                    spans += [(o, o + len(PAYLOADS[name])) for o in offs]
                inside = np.zeros(len(ra), bool)
                for lo, hi in spans:
                    inside[lo:hi] = True
                row.update(n_diff_bytes=int(idx.size), n_diff_outside_tmn=int(np.count_nonzero(x & ~inside)),
                           first_outside=[int(v) for v in np.flatnonzero(x & ~inside)[:10]], tmn_spans=spans)
            elif ra != rb:
                row["note"] = "length differs"
            pr.append(row)
        res["pairs"][f"{a}~{b}"] = pr
out_json.write_text(json.dumps(res, indent=1))
for k, v in res["blobs"].items():
    print(k, "records", v["n_records"], "lens", v["record_len"], "tmn hits", [{n: len(o) for n, o in h.items()} for h in v["tmn_hits"]])
for k, pr in res["pairs"].items():
    print(k, [{x: r.get(x) for x in ("record", "identical", "len_a", "len_b", "n_diff_bytes", "n_diff_outside_tmn", "note")} for r in pr])
