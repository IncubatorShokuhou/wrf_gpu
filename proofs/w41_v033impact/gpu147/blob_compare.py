"""Compare fused_d02 AOT blobs of the #147 arms (CPU, no GPU): are P (0227-compiled) and F/V (0707-compiled) the same
program beyond the baked TMN constant?

usage: JAX_PLATFORMS=cpu python blob_compare.py SNAP OUT_JSON LEAVES_0227.npz LEAVES_0707.npz LABEL=BLOB [LABEL=BLOB ...]
Per blob: format, sha256, native-record ELF (cubin) sha list, optimized-HLO text sha (+ the same with constant literals
masked). Per pair: blob byte diff ranges and whether every differing byte lies inside the TMN payload(s) (located by
searching each case's d02/d03 tbot bytes in the blob).
"""
import hashlib
import json
import re
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
TMN_KEYS = ("d02|noahmp_case.tbot", "d03|noahmp_case.tbot")
CONST = re.compile(r"constant\(\{[^}]*\}\)|constant\([^)]*\)")


def describe(path):
    b = Path(path).read_bytes()
    rec = {"path": path, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "format": ax._gpu_blob_format(b)}
    try:
        native = wcg._record(b, 2, ax, asm) if rec["format"] == ax.GPU_THUNK_FORMAT else b
        rec["elf_sha256"] = [hashlib.sha256(e).hexdigest()[:16] for e in wcg._elfs(native)]
        rec["native_record_bytes"] = len(native)
    except Exception as exc:  # noqa: BLE001
        rec["native_error"] = f"{type(exc).__name__}: {exc}"
    try:
        hlo = wcg._optimized_hlo(b, ax, asm)
        rec["opt_hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
        rec["opt_hlo_chars"] = len(hlo)
        rec["opt_hlo_const_masked_sha256"] = hashlib.sha256(CONST.sub("constant(C)", hlo).encode()).hexdigest()
        rec["opt_hlo_n_instr"] = hlo.count("\n")
    except Exception as exc:  # noqa: BLE001
        rec["hlo_error"] = f"{type(exc).__name__}: {exc}"
    hits = {}
    for case, lv in leaves.items():
        for k in TMN_KEYS:
            raw = np.ascontiguousarray(lv[k]).tobytes()
            for dt in (np.float32, np.float64):
                pay = np.ascontiguousarray(lv[k].astype(dt)).tobytes()
                pos, start = [], 0
                while len(pay) and (j := b.find(pay, start)) >= 0:
                    pos.append(j)
                    start = j + 1
                if pos:
                    hits[f"{case}:{k}:{np.dtype(dt).name}"] = {"offsets": pos, "len": len(pay)}
            del raw
    rec["tmn_payload_hits"] = hits
    return rec, b


def diff_ranges(a, b):
    n = min(len(a), len(b))
    x = np.frombuffer(a[:n], np.uint8) != np.frombuffer(b[:n], np.uint8)
    idx = np.flatnonzero(x)
    ranges = []
    if idx.size:
        start = prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i > prev + 64:
                ranges.append((start, prev + 1))
                start = i
            prev = i
        ranges.append((start, prev + 1))
    return ranges, int(idx.size)


res = {"blobs": {}, "pairs": {}}
raw = {}
for label, path in blobs.items():
    res["blobs"][label], raw[label] = describe(path)
labels = list(blobs)
for i, a in enumerate(labels):
    for b in labels[i + 1:]:
        A, B = raw[a], raw[b]
        ranges, ndiff = diff_ranges(A, B)
        tmn = [(o, o + h["len"]) for d in (res["blobs"][a], res["blobs"][b]) for h in d["tmn_payload_hits"].values() for o in h["offsets"]]
        outside = [r for r in ranges if not any(lo <= r[0] and r[1] <= hi for lo, hi in tmn)]
        ra, rb = res["blobs"][a], res["blobs"][b]
        res["pairs"][f"{a}~{b}"] = {
            "byte_identical": A == B, "same_length": len(A) == len(B), "n_diff_bytes": ndiff,
            "n_diff_ranges": len(ranges), "diff_ranges_outside_tmn_payloads": outside[:50],
            "n_outside": len(outside),
            "elf_identical": ra.get("elf_sha256") == rb.get("elf_sha256"),
            "opt_hlo_identical": ra.get("opt_hlo_sha256") == rb.get("opt_hlo_sha256"),
            "opt_hlo_identical_const_masked": ra.get("opt_hlo_const_masked_sha256") == rb.get("opt_hlo_const_masked_sha256")}
out_json.write_text(json.dumps(res, indent=1))
for k, v in res["pairs"].items():
    print(k, {x: v[x] for x in ("byte_identical", "same_length", "n_diff_bytes", "n_diff_ranges", "n_outside", "elf_identical",
                                 "opt_hlo_identical", "opt_hlo_identical_const_masked")})
for k, v in res["blobs"].items():
    print(k, v["bytes"], v.get("format"), "elfs", len(v.get("elf_sha256", [])), "hlo", v.get("opt_hlo_chars"), v.get("hlo_error"),
          "tmn hits", {h: len(x["offsets"]) for h, x in v["tmn_payload_hits"].items()})
