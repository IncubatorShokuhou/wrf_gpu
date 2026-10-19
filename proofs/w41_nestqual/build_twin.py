"""Build one original CPU-WRF twin of a G2 nested arm (lane w41-nestqual) — CPU only, no JAX, no ALISIOS reads.

Inputs come ONLY from the guarded stage written by stage_inputs.py (its STAGE.json); nothing under an ALISIOS root
is opened (asserted on every resolved path). Case layout = the v040 final-validation layout read by
proofs/v040/final_validation/score.py:
  <case>/input/namelist.input   arm namelist (= the GPU run's namelist) with ONLY the run length changed to the
                                D6 window (run_hours, end_*) and restart_interval raised (no restart files; numerics
                                unchanged); every other byte identical (PREPARE.json lists the exact line diff)
  <case>/input/<wrfinput/wrfbdy/wrflowinp>  symlinks to the staged copies (SHA-256 recorded = G2 arm inputs, E184)
  <case>/cpu/                   run dir: wrf.exe = the original WRF V4.7.1 dmpar build (sha 1d0f5779, the WN3
                                CPU-reference executable, also v-prep c42), the build's run tables (identical to
                                pristine 4.7.1 run/), Thompson .dat tables from a lane table dir if present (else WRF
                                computes them), inputs + namelist links
usage: python build_twin.py <STAGE.json> <cases_root> <arm> [--hours 24] [--minutes M (dry runs only)]
       [--tables DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

DMPAR = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/run")
DMPAR_SHA = "1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b"
ALISIOS_ROOTS = tuple(Path(p).resolve() for p in ("<DATA_ROOT>/server/work/src/alisios", "<DATA_ROOT>/server/work/alisios",
                                                  "<DATA_ROOT>/alisios"))
THOMPSON_TABLES = ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat")
SKIP = ("namelist.input", "wrf.exe", "real.exe", "ndown.exe", "tc.exe")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def not_alisios(path) -> Path:
    p = Path(path).resolve()
    assert not any(p == r or r in p.parents for r in ALISIOS_ROOTS), f"refusing an ALISIOS-root path: {p}"
    return p


def set_values(text: str, group: str, key: str, values) -> tuple[str, tuple[str, str]]:
    """Replace the value list of exactly one ``key`` inside ``&group`` keeping the line's alignment."""
    out, hit, inside = [], None, False
    for raw in text.split("\n"):
        s = raw.strip().lower()
        if s.startswith("&"):
            inside = s[1:] == group
        elif inside and s == "/":
            inside = False
        m = re.match(r"^(\s*" + re.escape(key) + r"\s*=\s*)(.*)$", raw, re.I) if inside else None
        if m:
            assert hit is None, f"{group}/{key} appears twice"
            new = m.group(1) + ", ".join(str(v) for v in values) + ","
            hit = (raw, new)
            raw = new
        out.append(raw)
    assert hit is not None, f"{group}/{key} not found"
    return "\n".join(out), hit


def values_of(text: str, key: str) -> list[str]:
    m = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", text, re.I | re.M)
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def twin_namelist(text: str, hours: int, minutes: int = 0, ranks: int = 4):
    ndom = int(values_of(text, "max_dom")[0])
    start = [datetime(*(int(values_of(text, f"start_{k}")[d]) for k in ("year", "month", "day", "hour")))
             for d in range(ndom)]
    assert len(set(start)) == 1, "nests must start with the root (G2 3-nest)"
    assert int(values_of(text, "run_days")[0]) == 0
    end = start[0] + timedelta(hours=hours)   # dry runs: run_minutes > 0 takes precedence over end_* in WRF
    diff = []
    edits = [("time_control", "run_hours", [hours]), ("time_control", "run_minutes", [minutes]),
             ("time_control", "end_year", [end.year] * ndom), ("time_control", "end_month", [end.month] * ndom),
             ("time_control", "end_day", [end.day] * ndom), ("time_control", "end_hour", [end.hour] * ndom),
             ("time_control", "restart_interval", [100000])]
    px, py = (int(values_of(text, k)[0]) for k in ("nproc_x", "nproc_y"))
    if px * py != ranks:   # decomposition only (WRF picks the factorization); no physics/dynamics option changes
        edits += [("domains", "nproc_x", [-1]), ("domains", "nproc_y", [-1])]
    for group, key, vals in edits:
        text, (old, new) = set_values(text, group, key, vals)
        if old != new:
            diff.append(dict(group=group, old=old.strip(), new=new.strip()))
    for key, want in (("history_interval", "60"), ("frames_per_outfile", "1")):
        assert values_of(text, key) == [want] * ndom, f"{key} must be {want} on every domain for D6 frames"
    return text, diff, start[0], end


def main():
    a = argparse.ArgumentParser()
    a.add_argument("stage", type=Path)
    a.add_argument("cases_root", type=Path)
    a.add_argument("arm")
    a.add_argument("--hours", type=int, default=24)
    a.add_argument("--minutes", type=int, default=0)
    a.add_argument("--tables", type=Path, default=None)
    a.add_argument("--name", default=None)
    a.add_argument("--ranks", type=int, default=4)
    args = a.parse_args()
    stage = json.loads(args.stage.read_text())
    assert stage["verdict"] == "OK", "stage receipt is not OK"
    nl = stage["namelists"][args.arm]
    assert nl["match"]
    case = args.cases_root / (args.name or f"{stage['issue']}_{args.arm}")
    assert not (case / "cpu_run.json").exists() and not list((case / "cpu").glob("wrfout_*")), \
        "never overwrite a reference"
    inp, cpu = case / "input", case / "cpu"
    inp.mkdir(parents=True, exist_ok=True)
    cpu.mkdir(exist_ok=True)
    src_nl = not_alisios(nl["staged"])
    assert sha256(src_nl) == nl["listed_sha256"] == nl["gpu_cli_input_namelist_sha256"]
    text, diff, start, end = twin_namelist(src_nl.read_text(), args.hours, args.minutes, args.ranks)
    (inp / "namelist.input").write_text(text)
    inputs = {}
    for name, row in stage["inputs"].items():
        src = not_alisios(row["staged"])
        digest = sha256(src)
        assert digest == row["sha256"] and row["prefix_match"], f"{name}: staged copy drifted"
        link = inp / name
        if link.is_symlink() or link.exists():
            assert link.resolve() == src, f"{link} points elsewhere"
        else:
            link.symlink_to(src)
        inputs[name] = dict(sha256=digest, staged=str(src), g2_store=f"{stage['input_dir']}/{name}",
                            arms_prefix=row["arms_prefix"])
    exe_sha = sha256(DMPAR / "wrf.exe")
    assert exe_sha == DMPAR_SHA, "WN3 reference executable drifted"
    links = {"wrf.exe": DMPAR / "wrf.exe"}
    links.update({p.name: p for p in DMPAR.iterdir() if p.is_file() and p.name not in SKIP})
    tables = {}
    if args.tables:
        for t in THOMPSON_TABLES:
            if (args.tables / t).is_file():
                links[t] = not_alisios(args.tables / t)
                tables[t] = sha256(args.tables / t)
    links.update({p.name: p for p in inp.iterdir()})
    for name, target in links.items():
        link = cpu / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(not_alisios(target) if name != "namelist.input" else target)
    prep = dict(schema="w41-nestqual.prepare.v1", case=case.name, issue=stage["issue"], arm=args.arm,
                stage_receipt=str(args.stage), stage_receipt_sha256=sha256(args.stage),
                arms_json_sha256=stage["arms_json_sha256"], gpu_arm_namelist_sha256=nl["gpu_cli_input_namelist_sha256"],
                arms_diff_vs_store=nl["diff_vs_store"], twin_namelist_sha256=sha256(inp / "namelist.input"),
                twin_namelist_diff_vs_arm=diff, window=dict(start=start.isoformat(), end=end.isoformat(),
                                                             hours=args.hours, minutes=args.minutes),
                inputs=inputs, thompson_tables=tables or "computed by WRF at init (none supplied)",
                executable=dict(path=str(DMPAR / "wrf.exe"), sha256=exe_sha, version="WRF V4.7.1 dmpar original",
                                ranks=args.ranks, note="WN3 CPU-reference build (D15/D26), v-prep c42; run tables == pristine 4.7.1"))
    (case / "PREPARE.json").write_text(json.dumps(prep, indent=1) + "\n")
    print(json.dumps(dict(case=str(case), diff=diff, window=prep["window"])))


if __name__ == "__main__":
    main()
