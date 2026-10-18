"""Replay the GPU OSSE smoke under the current continuous driver on the RECORDED member histories (CPU, no launch).

  python proofs/w41_optcont/replay_fix.py ROOT OUT.json

Proves that the critic-FIX changes (feasible prior bounds, precip refusal) leave G1-G5 unchanged: the driver re-derives
every member theta from the current code, the control files it writes must have the recorded content digests, every
member it asks for must already be in the ledger (prefetch launches nothing), and the replayed result must equal the
recorded result.json field by field.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__import__("os").environ.get("REPLAY_TREE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO/"src"))


def replay(root, out):
    from gpuwrf.optimize import continuous
    from gpuwrf.optimize.data import identity
    from gpuwrf.surface_params.controls import controls_digest, load_controls

    root = Path(root)
    run = root/"run"
    ledger = json.loads((run/"ledger.json").read_text())
    recorded = json.loads((run/"result.json").read_text())
    study = continuous.load_continuous(root/"study.json")
    asked = []

    def prefetch(pairs):
        for c, k in pairs:
            key = f"{c['id']}|{k['id']}"
            asked.append(key)
            if key not in ledger["members"]:
                raise SystemExit(f"replay would launch {key}: trajectory changed")

    def files(candidate, case):
        entry = ledger["members"][f"{candidate['id']}|{case['id']}"]
        paths = [Path(f["path"]) for f in entry["files"]]
        if [identity(p) for p in paths] != entry["files"]:
            raise SystemExit(f"member history changed: {candidate['id']}|{case['id']}")
        return paths, entry

    with tempfile.TemporaryDirectory(dir=root) as tmp:
        report = continuous.driver(study)(prefetch=prefetch, files=files, output=Path(tmp), ledger=ledger)
        digests = {}
        for path in sorted((Path(tmp)/"controls").glob("*.npz")):
            new, _ = load_controls(path)
            old, _ = load_controls(run/"controls"/path.name)
            digests[path.stem] = set(new) == set(old) and all(controls_digest(new[d]) == controls_digest(old[d]) for d in new)
        twin_same = (Path(tmp)/"twin"/"observations.csv").read_text() == (run/"twin"/"observations.csv").read_text()
    keys = ("estimate_multipliers", "posterior_sd_log", "ensemble_final_multipliers", "iterations", "evaluation", "twin",
            "program")
    replayed = json.loads(json.dumps(report, allow_nan=False))  # the recorded result went through the same JSON
    for r in (replayed, recorded):  # the twin observation file is bound by content; its path is the run's own dir
        r["twin"]["observations"].pop("path", None)
    Path(str(out)+".replayed.json").write_text(json.dumps({k: replayed[k] for k in keys}, indent=1, sort_keys=True)+"\n")
    same = {k: replayed[k] == recorded[k] for k in keys}
    est = {k: [replayed["estimate_multipliers"][k], recorded["estimate_multipliers"][k]] for k in recorded["estimate_multipliers"]}
    bounds = {q["name"]: {"bounds": q["bounds"], "feasible_factors": q.get("feasible_factors"), "source": q.get("bounds_source")}
              for q in study["parameters"]}
    summary = {"scope": "CPU replay of the recorded GPU OSSE member histories under the critic-FIX code (no launch)",
               "source": continuous.__file__,
               "bounds_now": bounds, "members_asked": len(asked), "members_missing": 0,
               "controls_digest_equal": all(digests.values()), "controls_compared": len(digests),
               "twin_observations_equal": twin_same, "result_fields_equal": same,
               "estimate_replayed_vs_recorded": est,
               "replayed_fields_sha256": __import__("hashlib").sha256(
                   json.dumps({k: replayed[k] for k in keys}, sort_keys=True).encode()).hexdigest(),
               "blas_threads_env": {k: __import__("os").environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
               "all_equal": all(same.values()) and all(digests.values()) and twin_same,
               "recorded_result_sha256": identity(run/"result.json")["sha256"]}
    Path(out).write_text(json.dumps(summary, indent=1)+"\n")
    print(json.dumps({k: summary[k] for k in ("all_equal", "members_asked", "controls_compared", "result_fields_equal")}))
    return 0 if summary["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(replay(*sys.argv[1:]))
