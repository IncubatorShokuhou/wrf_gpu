"""Drive compile_one over a capture_real.py directory: every unique kernel x target (resumable).

usage: compile_matrix.py CAPTURE_DIR OUT.jsonl TARGETS   e.g. cuda:70,cuda:75,hip:gfx942
env: O5_WORKERS (3), O5_TIMEOUT (1200 s per compile), O5_MAX_CHARS (0 = all)
Smallest kernels first per target; a timeout is recorded as TIMEOUT (never a PASS).
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys

cap, outp, targets = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
workers = int(os.environ.get("O5_WORKERS", "3"))
tmo = int(os.environ.get("O5_TIMEOUT", "1200"))
maxc = int(os.environ.get("O5_MAX_CHARS", "0"))
rec = json.load(open(os.path.join(cap, "receipt.json")))
kernels = [k for k in rec["kernels"] if "ttir_sha16" in k]
platform = rec["platform"]
done = set()
if os.path.exists(outp):
    for line in open(outp):
        r = json.loads(line)
        done.add((r["ttir"], r["backend"], r["arch"], r["num_warps"]))
jobs = []
for t in targets:
    be, arch = t.split(":")
    if (platform == "rocm") != (be == "hip"):
        continue
    for k in kernels:
        fn = f"{platform}_{k['ttir_sha16']}.ttir"
        nw = k.get("num_warps") or 4
        ns = k.get("num_stages") or (1 if be == "hip" else 3)
        if (fn, be, arch, nw) in done or (maxc and k["ttir_chars"] > maxc):
            continue
        jobs.append((k["ttir_chars"], fn, be, arch, nw, ns, k))
jobs.sort(key=lambda j: j[0])
print(len(jobs), "jobs", flush=True)
here = os.path.dirname(os.path.abspath(__file__))


def run(job):
    _, fn, be, arch, nw, ns, k = job
    meta = {"name": k["name"], "src": k["src"].split("/src/gpuwrf/")[-1], "ttir_chars": k["ttir_chars"],
            "calls": k["calls"], "grid": k["grid"]}
    try:
        r = subprocess.run([sys.executable, os.path.join(here, "compile_one.py"), os.path.join(cap, "ttir", fn),
                            be, arch, str(nw), str(ns)], capture_output=True, text=True, timeout=tmo)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        res = json.loads(lines[-1]) if lines else {"ttir": fn, "backend": be, "arch": arch, "num_warps": nw,
                                                     "status": "CRASH", "error": (r.stderr or "")[-400:]}
    except subprocess.TimeoutExpired:
        res = {"ttir": fn, "backend": be, "arch": arch, "num_warps": nw, "status": "TIMEOUT", "timeout_s": tmo}
    res.update(meta)
    return json.dumps(res)


with cf.ThreadPoolExecutor(workers) as ex, open(outp, "a") as f:
    for line in ex.map(run, jobs):
        f.write(line + "\n")
        f.flush()
        print(line[:220], flush=True)
