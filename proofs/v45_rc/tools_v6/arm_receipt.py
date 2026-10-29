"""v45-rc: post-arm receipt (stdlib only, no JAX/gpuwrf import; content hashes only, no st_dev/st_ino).

usage: python3 -I arm_receipt.py RUN_ROOT ARM_DIR TIMER_JSON RECEIPT_DIR [--delete-wrfout]
- rc.txt / t_end_unix are filled from the arm_timer receipt when the arm was killed before arm.sh wrote them.
- mtimes.json = ORIGINAL wrfout mtimes (E162, written before anything else touches the files), wrfout_sha.txt
  (sha256sum format), WRFOUT_FILES.tsv (name sha256 bytes mtime), PROOF_SUMMARY.json (walls, verdict, finiteness, AOT/prefetch
  sources) and RECEIPT.json; then the small files are copied to RECEIPT_DIR/<arm>/.
- --delete-wrfout: after the SHA list is written and re-read, unlink exactly the wrfout_d0* regular files that are in it
  (inside ARM_DIR/wrfout, which must lie inside RUN_ROOT) and remove the then-empty wrfout dir. Never anything else.
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

run_root, arm, timer_json, rdir = (Path(x) for x in sys.argv[1:5])
delete = "--delete-wrfout" in sys.argv[5:]
assert arm.resolve().is_relative_to(run_root.resolve()), (arm, run_root)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


timer = load(timer_json) or {}
if not (arm / "rc.txt").exists() and "rc" in timer:
    (arm / "rc.txt").write_text(f"{timer['rc']}\n")
if not (arm / "t_end_unix").exists() and "end_unix" in timer:
    (arm / "t_end_unix").write_text(f"{timer['end_unix']}\n")
wdir = arm / "wrfout"
files = sorted(p for p in wdir.glob("wrfout_d0*") if p.is_file() and not p.is_symlink()) if wdir.is_dir() else []
mt = {p.name: p.stat().st_mtime for p in files}
(arm / "mtimes.json").write_text(json.dumps(mt, indent=1) + "\n")
rows = [(p.name, sha(p), p.stat().st_size, mt[p.name]) for p in files]
(arm / "wrfout_sha.txt").write_text("".join(f"{s}  {n}\n" for n, s, _, _ in rows))
(arm / "WRFOUT_FILES.tsv").write_text("name\tsha256\tsize\tmtime\n" + "".join(f"{n}\t{s}\t{b}\t{m}\n" for n, s, b, m in rows))
proof = load(arm / "proof/nested_pipeline_run.json") or {}
meta = proof.get("metadata") or {}
aot_lines = re.findall(r"\[gpuwrf:nested-aot\][^\n]*", (arm / "run.log").read_text(errors="replace")) if (arm / "run.log").exists() else []
summary = dict(verdict=proof.get("verdict"), all_domains_finite=proof.get("all_domains_finite"),
               all_outputs_present=proof.get("all_outputs_present"), hours=proof.get("hours"),
               wall_clock_total_s=proof.get("wall_clock_total_s"), wall_clock_forecast_only_s=proof.get("wall_clock_forecast_only_s"),
               aot_prefetch_elapsed_s=(meta.get("aot_prefetch") or {}).get("elapsed_s"),
               nested_aot_programs=sorted(((meta.get("nested_aot") or {}).get("domains") or {})),
               host_suspend=proof.get("host_suspend"), device=proof.get("device"), n_aot_log_lines=len(aot_lines),
               aot_log_lines=aot_lines[:40])
(arm / "PROOF_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", (arm / "time.txt").read_text()) if (arm / "time.txt").exists() else None
rec = dict(arm=arm.name, rc=(arm / "rc.txt").read_text().strip() if (arm / "rc.txt").exists() else None, timer=timer,
           n_wrfout=len(rows), wrfout_bytes=sum(r[2] for r in rows), wrfout_sha_list_sha256=sha(arm / "wrfout_sha.txt"),
           max_rss_gib=round(int(rss.group(1)) / 2**20, 3) if rss else None, proof=summary, deleted_wrfout=False)
if delete and rows:
    listed = {l.split()[1] for l in (arm / "wrfout_sha.txt").read_text().splitlines() if l.strip()}
    assert listed == {r[0] for r in rows}
    for name in sorted(listed):
        p = wdir / name
        assert p.parent == wdir and p.is_file() and not p.is_symlink()
        p.unlink()
    if not any(wdir.iterdir()):
        wdir.rmdir()
    rec["deleted_wrfout"] = True
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import registry_hook
    registry_hook.record_deleted(arm)
(arm / "RECEIPT.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
out = rdir / arm.name
out.mkdir(parents=True, exist_ok=True)
for n in ("REGISTRY_START.json", "REGISTRY_END.json", "device_measured.json", "CPU_SMOKE_BOUNDED.json", "CPU_PLAN_ONLY.json", "CPU_PROFILE_SUBSTITUTION.json", "PROFILE_CALLS.json", "RECEIPT.json", "PROOF_SUMMARY.json", "WRFOUT_FILES.tsv", "wrfout_sha.txt", "mtimes.json", "rc.txt", "t0_unix",
          "t_end_unix", "time.txt", "device.txt", "env_product.txt", "env_child.txt", "command.txt", "source_head.txt",
          "check_before.txt", "check_after.txt", "core_watch_summary.json", "core_watch_strict_summary.json",
          "sched_watch_summary.json", "ABORT", "timer.json"):
    if (arm / n).is_file():
        shutil.copy2(arm / n, out / n)
if (arm / "proof/nested_pipeline_run.json").is_file():
    shutil.copy2(arm / "proof/nested_pipeline_run.json", out / "nested_pipeline_run.json")
if (arm / "run.log").is_file():
    (out / "run.log.tail").write_text("".join((arm / "run.log").read_text(errors="replace").splitlines(True)[-60:]))
print(json.dumps({k: rec[k] for k in ("arm", "rc", "n_wrfout", "wrfout_bytes", "max_rss_gib", "deleted_wrfout")}))
