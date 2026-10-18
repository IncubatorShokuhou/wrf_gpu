"""Composed default requal, CPU follow-up after the lock (E41 before any cache cleanup, E166/E177).

usage: JAX_PLATFORMS=cpu python followup.py ROOT [--skip-d6] [--skip-e41] [--skip-ids]
Writes ROOT/RESULT.json:
  per case (Swiss / PROD / WN3-0227): model rc, wall, frames, E41 = scripts/wave_cubin_gate.py of the candidate on
    the case's actual cache (every rich JIT + AOT domain/fused program; the PROD cache also holds the S1 programs),
    D6 per domain = SHA-pinned F20 scorer (frozen manifest, full CPU inventory, nonfinite) vs original CPU-WRF;
  S1 = scripts/bench_summary.py of the candidate over the harness receipts (prime as cold, warm1, warm2):
    s1_warm_s_per_fc_h (seg 2-3), A/A, vs RC2 3.088 / LW9 5.633;
  identities = proofs/s1_port/release_identity.py of the candidate (Swiss RD11, PROD s0_case_20260725).
Missing evidence is reported as missing, never as PASS.
"""
import hashlib, json, os, re, subprocess, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
PINS = {"score.py": "4c025af8382c9626e7f7e0de1e888a91bd0270376f5c837431d1000c70c2c21d",
        "score_with_handles.py": "7407b0ddc611640b318e930f29b62673015f894fbc39d76f34d42856f92d8d19"}


def identity_sha(path):
    """Default-program identity of a release_identity.py receipt: arms.release.sha (64 hex), else MISSING/INVALID."""
    path = Path(path)
    if not path.exists():
        return "MISSING"
    try:
        value = json.loads(path.read_text())["arms"]["release"]["sha"]
    except (ValueError, KeyError, TypeError):
        return "INVALID"
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else "INVALID"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rc_of(log):
    found = re.findall(r"arm rc=(\d+)", Path(log).read_text()) if Path(log).exists() else []
    return int(found[-1]) if found else None


def wall_of(log):
    found = re.findall(r"Elapsed \(wall clock\) time .*: (\S+)", Path(log).read_text()) if Path(log).exists() else []
    return sum(float(v) * 60 ** i for i, v in enumerate(reversed(found[-1].split(":")))) if found else None


if __name__ == "__main__":
    ROOT = Path(sys.argv[1]).resolve()
    SKIP = set(sys.argv[2:])
    plan = json.loads((ROOT / "PLAN.json").read_text())
    SNAP = Path(plan["candidate"])
    ENV = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(SNAP / "src"))
    for name, pin in PINS.items():
        assert sha(ROOT / "tools" / name) == pin, f"scorer pin {name}"
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=SNAP, capture_output=True, text=True).stdout.strip() == \
        plan["candidate_head"], "snapshot moved"
    result = {"plan": plan, "cases": {}, "S1": None, "identities": {}}
    for case_name, arm in plan["arms"].items():
        case = Path(arm["case"])
        log = case / "run24/run.log"
        shas = case / "run24/wrfout.sha256"
        row = {"rc": rc_of(log), "wall_s": wall_of(log),
               "frames": len(shas.read_text().split("\n")) - 1 if shas.exists() else 0, "expected_frames": arm["cpu_frames"]}
        if "--skip-e41" not in SKIP and not (case / "e41/wave_gate.json").exists():
            command = [sys.executable, str(SNAP / "scripts/wave_cubin_gate.py"), str(case / "cache/jax"), str(case / "e41")]
            proof = case / "run24/proof/nested_pipeline_run.json"
            if proof.exists():
                command += ["--run-proof", str(proof)]
            subprocess.run(command, cwd=SNAP, env=ENV, timeout=2400)
        if (case / "e41/wave_gate.json").exists():
            gate = json.loads((case / "e41/wave_gate.json").read_text())
            row["E41"] = {"gate": gate.get("e41_gate"), "coverage": gate.get("coverage", {}).get("status")}
        row["D6"] = {}
        for domain in arm["domains"]:
            target = case / f"D6_{domain}.json"
            if "--skip-d6" not in SKIP and not target.exists() and row["rc"] == 0:
                subprocess.run([sys.executable, str(ROOT / "tools/score_with_handles.py"), str(SNAP), str(case),
                                "--domains", domain], cwd=SNAP, env=ENV, timeout=3600)
            if target.exists():
                d6 = json.loads(target.read_text())
                row["D6"][domain] = {k: d6.get(k) for k in ("verdict", "paired_frames", "coverage_complete")} | {
                    "bounded_misses": len(d6.get("failures", [])), "nonfinite": len(d6.get("nonfinite", []))}
            else:
                row["D6"][domain] = "MISSING"
        result["cases"][case_name] = row

    s1 = Path(plan["s1"]["root"])
    receipts = {label: s1 / arm / "run/receipt.json" for label, arm in (("cold", "prime"), ("warm1", "warm1"), ("warm2", "warm2"))}
    present = {k: v for k, v in receipts.items() if v.exists()}
    if present:
        out = s1 / "bench.json"
        subprocess.run([sys.executable, str(SNAP / "scripts/bench_summary.py"), "--out", str(out), "--tag", "w41bdyrec_composed",
                        "--cache", plan["s1"]["cache"], "--src", str(SNAP), "--src-rev", plan["candidate_head"],
                        "--src-tree", plan["candidate_src_tree"], "--xla-flags="] + [f"{k}={v}" for k, v in present.items()],
                       cwd=SNAP, env=ENV, timeout=600)
        if out.exists():
            bench = json.loads(out.read_text())
            value = bench.get("s1_warm_s_per_fc_h")
            result["S1"] = {"bench_json": str(out), "s1_warm_s_per_fc_h": value, "headline_valid": bench.get("headline_valid"),
                            "reject_reasons": bench.get("reject_reasons"), "aa": bench.get("aa"),
                            "vs_baseline": {k: (round(value / v, 4) if value else None) for k, v in plan["s1"]["baseline"].items()
                                            if isinstance(v, float)}, "baseline": plan["s1"]["baseline"]}
    result["S1_arms_missing"] = sorted(set(receipts) - set(present))

    if "--skip-ids" not in SKIP:
        for name, extra in (("swiss", []), ("prod", ["<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"])):
            out = ROOT / f"identity_{name}.json"
            if not out.exists():
                subprocess.run([sys.executable, str(SNAP / "proofs/s1_port/release_identity.py"), str(SNAP), str(out), *extra],
                               cwd=SNAP, env=dict(ENV, OMP_NUM_THREADS="1"), timeout=2400)
            result["identities"][name] = identity_sha(out)
    (ROOT / "RESULT.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("cases", "S1", "S1_arms_missing", "identities")}, indent=1, default=str))
