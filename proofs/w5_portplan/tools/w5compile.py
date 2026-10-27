"""Drive w5compile_one over a capture dir: every unique TTIR x targets, N workers, per-job timeout."""
import json, os, subprocess, sys, concurrent.futures as cf
cap, outp, targets = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
workers = int(os.environ.get("W5_WORKERS", "3")); tmo = int(os.environ.get("W5_TIMEOUT", "300"))
meta = {}
for line in open(os.path.join(cap, "index.jsonl")):
    m = json.loads(line)
    if m.get("status") == "LOWERED":
        meta[f"{m['platform']}_{m['ttir_sha16']}.ttir"] = m
done = set()
if os.path.exists(outp):
    for line in open(outp):
        r = json.loads(line); done.add((r["ttir"], r["arch"]))
jobs = []
for fn, m in sorted(meta.items()):
    for t in targets:
        be, arch = t.split(":")
        if (m["platform"] == "rocm") != (be == "hip"): continue
        if (fn, arch) in done: continue
        nw = m.get("num_warps") or 4
        ns = m.get("num_stages") or (1 if be == "hip" else 3)
        jobs.append((fn, be, arch, nw, ns))
order = {t: i for i, t in enumerate(targets)}
jobs.sort(key=lambda j: (order[f"{j[1]}:{j[2]}"], meta[j[0]]["ttir_chars"]))
maxc = int(os.environ.get("W5_MAX_CHARS", "0"))
if maxc: jobs = [j for j in jobs if meta[j[0]]["ttir_chars"] <= maxc]
print(len(jobs), "jobs", flush=True)
def run(j):
    fn, be, arch, nw, ns = j
    try:
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "w5compile_one.py"), os.path.join(cap, fn), be, arch, str(nw), str(ns)],
                           capture_output=True, text=True, timeout=tmo)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        return lines[-1] if lines else json.dumps({"ttir": fn, "backend": be, "arch": arch, "status": "CRASH", "error": (r.stderr or "")[-400:]})
    except subprocess.TimeoutExpired:
        return json.dumps({"ttir": fn, "backend": be, "arch": arch, "status": "TIMEOUT", "timeout_s": tmo})
with cf.ThreadPoolExecutor(workers) as ex, open(outp, "a") as f:
    for res in ex.map(run, jobs):
        f.write(res + "\n"); f.flush(); print(res[:200], flush=True)
