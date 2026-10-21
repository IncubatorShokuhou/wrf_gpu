#!/usr/bin/env python3
"""v2 (top-up): GROUPS = {"topup": rw3t_topup/runs}; no keep rules (every finished arm incl. O72 is deleted after its
receipt; byte compares use wrfout_sha.txt). Otherwise identical to v1.
w43-release disk relief (manager DISK URGENT 2026-10-21 ~14:30Z; ALISIOS floor >= 200 GiB). Separate process: never touches
the running GPU session files (E163). For every FINISHED arm (wrfout_sha.txt complete = one line per wrfout file on disk,
newline-terminated, unchanged for >= 30 s; arm.sh writes rc.txt -> mtimes.json -> wrfout_sha.txt in that order):
  1. receipt  $L/arm_receipts/<group>/<arm>/WRFOUT_FILES.tsv : name, sha256 (arm.sh post-run hash), size, mtime, Times;
     verifies every file is unchanged since that hash (name set == wrfout_sha.txt == mtimes.json, mtime == mtimes.json) — no
     re-hash (would read ~50 GB inside QUIET timing windows); copies wrfout_sha.txt, mtimes.json, rc.txt, check_*.txt, t*_unix.
  2. delete the arm's wrfout files by exact name (names from the receipt, pattern-checked), then rmdir wrfout/.
     Only while /tmp/wrf_gpu2_quiet is ABSENT (our own RW3T timing arms set it); resumes in the next gap if QUIET appears.
     One MAILBOX deletion line per arm.
Keep: RW3T N72 (until the D6 decision). A3 is deleted only after N72 finished with a sha list == A3's (a 0227 72 h copy stays).
Floor: if <DATA_ROOT> free < 215 GiB while the session runs -> post + TERM the gpu_session.sh PID (its trap ends cleanly).
usage: disk_relief_v1.py <gpu_session_pid>   (exits when the session is gone and every eligible arm is done)
"""
import json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
import netCDF4

L = Path("<USER_HOME>/wrf_gpu2_lanes/w43-release"); S = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release")
GROUPS = {"topup": S / "rw3t_topup/runs"}
QUIET = Path("/tmp/wrf_gpu2_quiet"); MAILBOX = Path("<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md")
REAL = "<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh"; FLOOR_G = 215
NAME = re.compile(r"^wrfout_d0[1-9]_\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d$")
SPID = int(sys.argv[1]); LOG = L / "disk_relief_topup.log"


def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def log(m):
    with open(LOG, "a") as f: f.write(f"{utc()} {m}\n")
def post(m):
    real = subprocess.run([REAL, "--iso"], capture_output=True, text=True).stdout.strip()
    with open(MAILBOX, "a") as f: f.write(f"{utc()} w43-release: [real {real}] {m}\n")
    log("POST " + m)
def free_g(): return shutil.disk_usage("<DATA_ROOT>").free // 2**30
def alive(): return Path(f"/proc/{SPID}").exists()
def shas(p): return {l.split()[-1]: l.split()[0] for l in p.read_text().splitlines() if l.strip()}


def finished(d):
    s, m, w = d / "wrfout_sha.txt", d / "mtimes.json", d / "wrfout"
    if not (s.exists() and m.exists() and w.is_dir()) or time.time() - s.stat().st_mtime < 30: return False
    txt = s.read_text()
    return txt.endswith("\n") and len(shas(s)) == len(list(w.iterdir()))


def receipt(group, d):
    sha = shas(d / "wrfout_sha.txt"); mt = json.loads((d / "mtimes.json").read_text()); w = d / "wrfout"
    on_disk = sorted(p.name for p in w.iterdir())
    bad = [n for n in set(on_disk) | set(sha) if not NAME.match(n)]
    if bad or sorted(sha) != on_disk or sorted(mt) != on_disk:
        return None, f"name sets differ/unexpected (bad {sorted(bad)[:3]}, sha {len(sha)}, mt {len(mt)}, disk {len(on_disk)})"
    rows = []
    for n in on_disk:
        st = (w / n).stat()
        if st.st_mtime != mt[n]: return None, f"{n} mtime changed since hash ({st.st_mtime} vs {mt[n]})"
        with netCDF4.Dataset(w / n) as ds:
            times = ";".join(str(x) for x in netCDF4.chartostring(ds.variables["Times"][:]).reshape(-1))
        rows.append((n, sha[n], st.st_size, repr(st.st_mtime), times))
    out = L / "arm_receipts" / group / d.name; out.mkdir(parents=True, exist_ok=True)
    with open(out / "WRFOUT_FILES.tsv", "w") as f:
        f.write("name\tsha256\tsize\tmtime\tTimes\n"); f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    for n in ("wrfout_sha.txt", "mtimes.json", "rc.txt", "check_before.txt", "check_after.txt", "t0_unix", "t_end_unix"):
        if (d / n).exists(): shutil.copy2(d / n, out / n)
    return rows, out


def delete(group, d, rows, out):
    w = d / "wrfout"
    for n, *_ in rows:
        if QUIET.exists(): return False
        if (w / n).exists(): os.unlink(w / n)
    os.rmdir(w)
    nbytes = sum(r[2] for r in rows)
    post(f"DISK deleted {group}/{d.name} wrfouts: {len(rows)} files {nbytes / 1e9:.2f} GB by exact name (receipt "
         f"{out}/WRFOUT_FILES.tsv: sha256+size+mtime+Times; byte compares use wrfout_sha.txt, kept); <DATA_ROOT> free {free_g()} GiB")
    return True


def eligible(group, d):
    if not d.is_dir() or "aborted" in d.name: return False
    return finished(d)


log(f"START session pid {SPID} alive={alive()} free {free_g()} GiB")
done, refused, receipts, floor_hit = set(), set(), {}, False
while True:
    if alive() and free_g() < FLOOR_G and not floor_hit:
        floor_hit = True
        post(f"DISK FLOOR: <DATA_ROOT> free {free_g()} GiB < {FLOOR_G} -> TERM gpu_session.sh {SPID} (no further RW3T arm)")
        os.kill(SPID, signal.SIGTERM)
    pending = [(g, d) for g, root in GROUPS.items() if root.is_dir() for d in sorted(root.iterdir())
               if f"{g}/{d.name}" not in done | refused and (f"{g}/{d.name}" in receipts or eligible(g, d))]
    for g, d in pending:
        if QUIET.exists(): break
        key = f"{g}/{d.name}"
        if key not in receipts:
            rows, out = receipt(g, d)
            if rows is None:
                refused.add(key); post(f"DISK NOT deleted {key}: {out}"); continue
            receipts[key] = (rows, out); log(f"RECEIPT {key} {len(rows)} files -> {out}")
        if delete(g, d, *receipts[key]): done.add(key)
        else: log(f"PAUSE {key}: QUIET appeared mid-delete (resume next gap)")
    if not alive() and not pending: break
    time.sleep(10)
log(f"END done {sorted(done)} refused {sorted(refused)} free {free_g()} GiB")
