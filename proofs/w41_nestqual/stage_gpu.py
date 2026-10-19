"""Guarded staging of one G2 GPU arm's output for the D6 nest gate (lane w41-nestqual) — CPU only, no JAX.

G2 was paused on 18.10 (v0.4.0 fused_d02 AOT-key regression) and reruns the arms on 19.10 08-13Z with gpuwrf v0.4.1
(129248b4, bundle 24a7b24f); rd-s6 keeps the raw wrfout of issue 20260707_18z read-only (KEEP_RAW) under
g2/runs/<issue>/<arm>/<tag>/ (tag v041; v040 for the paused 18.10 runs) and lists every file's SHA-256 in its manifest
<rd-s6 state>/g2/G2_<MMDD>_OUTPUTS_MANIFEST_<tag>.json (cross TO_WRF_GPU2_G2_0707_OUTPUTS_READY_20261019.md).
This copies, inside ONE sealed alisios.guarded_read session (scan -> admit -> READ_SPY, plus our refusing spy on every
ALISIOS root), exactly:
  * wrfout_d01..d03 frames with lead 0..<hours> h (WRF alarm stamps such as 01:00:18 included) -> D6 window
  * the GPU run's receipts: proof/nested_pipeline_run.json, run.log, env.txt, source_head.txt, lock.log
  * the arm's live in/namelist.input (its SHA must still equal the namelist staged for the twin, E184)
  * the rd-s6 output manifest when it exists (every staged frame's SHA must equal its manifest entry)
and links them into <case>/gpu (+ <case>/gpu_receipts) of the CPU twin case built by build_twin.py. The directory
listing (file names only) is metadata; no payload byte is read outside the session.
Days: issue day + every valid day of the copied frames/receipts (0708..0710 for a 48-h run from 0708 00Z); the guard
is scanned on issue day .. +4 first (non-ALLOW = refuse everything before any open).
usage: JAX_PLATFORMS=cpu PYTHONPATH=<tree>/src python stage_gpu.py <case_dir> <stage_root> <receipt_dir> [hours=24] [tag=v041]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from gpuwrf.assimilation.letkf import alisios_guard as ag

G2 = Path("<DATA_ROOT>/alisios/scratch/rd-s6-wrf-cloud-attribution/g2")
RD = Path("<DATA_ROOT>/alisios/state/manager/runs/mgr_wave_20261009/rd-s6-wrf-cloud-attribution")
RECEIPTS = ("proof/nested_pipeline_run.json", "run.log", "env.txt", "source_head.txt", "lock.log")
NAME = re.compile(r"^wrfout_(d0[1-9])_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)$")
PURPOSE = "wrf_gpu2 w41-nestqual: G2 GPU arm wrfout (0-24 h) + run receipts for the D6 nest gate vs original CPU-WRF"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(case, stage_root, receipt_dir, hours=24, tag="v041"):
    case, stage_root, receipt_dir = Path(case).resolve(), Path(stage_root), Path(receipt_dir)
    prep = json.loads((case / "PREPARE.json").read_text())
    issue, arm = prep["issue"], prep["arm"]
    start = datetime.fromisoformat(prep["window"]["start"])
    root_dt = 54.0
    run = G2 / "runs" / issue / arm / tag
    in_dir = G2 / "runs" / issue / arm / "in"
    manifest = RD / "g2" / f"G2_{issue[4:8]}_OUTPUTS_MANIFEST_{tag}.json"
    names = sorted(p.name for p in (run / "wrfout").iterdir())          # listing only (metadata)
    frames = {}
    for n in names:
        m = NAME.match(n)
        if not m:
            continue
        lead = (datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S") - start).total_seconds()
        if 0 <= lead < hours * 3600 + root_dt + .01:
            frames.setdefault(m.group(1), []).append(n)
    assert sorted(frames) == ["d01", "d02", "d03"], f"GPU domains {sorted(frames)}"
    assert all(len(v) == hours + 1 for v in frames.values()), {d: len(v) for d, v in frames.items()}
    issue_day = issue[:8]
    span = [(datetime.strptime(issue_day, "%Y%m%d") + timedelta(days=k)).strftime("%Y%m%d") for k in range(5)]
    _, verdicts = ag.drop_deny_days([(d, d) for d in span])
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "guard_scan.json").write_text(json.dumps(dict(mode=ag.MODE, days=span, verdicts=verdicts), indent=1) + "\n")
    deny = {d: v for d, v in verdicts.items() if v != "ALLOW"}
    if deny or set(verdicts) != set(span):
        raise SystemExit(f"GUARD REFUSED {deny or verdicts}: nothing opened, stop and report")
    run_days = sorted({(start + timedelta(hours=h)).strftime("%Y%m%d") for h in range(0, 49)})
    targets = {}
    for d, files in frames.items():
        for n in files:
            targets[str(run / "wrfout" / n)] = ([issue_day], [n[11:21].replace("-", "")])
    for r in RECEIPTS:
        targets[str(run / r)] = ([issue_day], run_days)
    targets[str(in_dir / "namelist.input")] = ([issue_day], [issue_day])
    if manifest.exists():                                                 # existence = metadata
        targets[str(manifest)] = ([issue_day], [issue_day])
    staged, receipt = ag.facade_stage(targets, stage_root, receipt_dir / "payload", purpose=PURPOSE)
    gpu, rec = case / "gpu", case / "gpu_receipts"
    gpu.mkdir(exist_ok=True)
    rec.mkdir(exist_ok=True)
    assert not list(gpu.iterdir()) and not list(rec.iterdir()), "gpu dirs must be empty (fresh staging per attempt)"
    out = {}
    for p, s in staged.items():
        name = Path(p).name
        dst = gpu / name if NAME.match(name) else rec / name
        dst.symlink_to(Path(s).resolve())
        out[name] = dict(source=p, staged=s, sha256=receipt["files"][p]["sha256"], size=receipt["files"][p]["size"])
    # E184: the GPU process's own --input-dir: its link targets (metadata) must be the store files whose staged
    # copies carry the ARMS SHA prefixes, and its namelist the staged arm namelist; proof records that input_dir
    stage = json.loads(Path(prep["stage_receipt"]).read_text())
    links = {f: str(Path(in_dir / f).resolve()) for f in stage["inputs"]}
    identity = dict(input_dir=str(in_dir),
                    links_match_store={f: links[f] == str((Path(stage["input_dir"]) / f).resolve()) for f in links},
                    staged_input_sha256={f: r["sha256"] for f, r in stage["inputs"].items()},
                    gpu_namelist_sha256=stage["namelists"][arm]["gpu_cli_input_namelist_sha256"],
                    gpu_namelist_restaged_sha256=out["namelist.input"]["sha256"],
                    proof_input_dir=json.loads((rec / "nested_pipeline_run.json").read_text()).get("input_dir"))
    om = None
    if manifest.name in out:          # rd-s6 SHA manifest: every staged frame must be the file rd-s6 listed
        m = json.loads((rec / manifest.name).read_text())
        listed = m["arms"][arm]["files"]
        frames_out = [n for n in out if NAME.match(n)]
        om = dict(path=str(manifest), sha256=out[manifest.name]["sha256"], engine=m.get("engine"),
                  status=m["arms"][arm]["status"], partial=m.get("partial"), listed_files=len(listed),
                  matched=sum(listed.get(n, {}).get("sha256") == out[n]["sha256"] for n in frames_out),
                  mismatched=sorted(n for n in frames_out if n in listed and listed[n]["sha256"] != out[n]["sha256"]),
                  not_listed=sorted(n for n in frames_out if n not in listed))
        om["announced_sha256_prefix"] = os.environ.get("NESTQUAL_MANIFEST_SHA", "")   # rd-s6 cross note / manager GO
        om["ok"] = (om["status"] == "DONE_CLEAN" and om["matched"] == len(frames_out)
                    and om["sha256"].startswith(om["announced_sha256_prefix"]))
    identity["output_manifest"] = om
    identity["ok"] = (all(identity["links_match_store"].values())
                      and identity["proof_input_dir"] in (str(in_dir), str(in_dir.resolve()))
                      and identity["gpu_namelist_restaged_sha256"] == identity["gpu_namelist_sha256"]
                      == prep["gpu_arm_namelist_sha256"]
                      and (om is None or om["ok"]))
    env = (rec / "env.txt").read_text(errors="replace").splitlines()
    engine = dict(source_head=(rec / "source_head.txt").read_text().strip(),
                  pythonpath=next((ln.split("=", 1)[1] for ln in env if ln.startswith("PYTHONPATH=")), None),
                  manifest_engine=om["engine"] if om else None)
    log = (rec / "run.log").read_text(errors="replace")
    arm_rc = re.findall(r"arm rc=(\d+)", log)
    result = dict(schema="w41-nestqual.stage_gpu.v1", case=case.name, issue=issue, arm=arm, hours=hours,
                  gpu_run_dir=str(run), engine=engine, input_identity=identity, frames={d: len(v) for d, v in frames.items()}, files=out,
                  gpu_arm_rc=arm_rc[-1] if arm_rc else None, aot_loaded_true=log.count("loaded=true"),
                  aot_fallback_missing=log.count("fallback:missing"), guard_verdicts=verdicts,
                  facade_receipt=str(receipt_dir / "payload" / "facade_stage.json"),
                  spy_refused=receipt["spy"]["refused_paths"], counters=receipt["counters"])
    (case / "GPU_STAGE.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("case", "frames", "gpu_arm_rc", "aot_loaded_true", "aot_fallback_missing",
                                              "spy_refused", "counters")}))
    return 0 if result["gpu_arm_rc"] == "0" and identity["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]), *sys.argv[5:6]))
