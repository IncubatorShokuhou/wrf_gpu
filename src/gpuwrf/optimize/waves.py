"""Host-only wave checkpoints and external review tokens; no model/backend imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


class Waves:
    """One atomic cursor consumes a GO and preserves the exact post-wave driver state.

    A GO is an external review assertion, not authentication of the writer. Filesystem
    permissions must restrict writes to the operator/reviewer. Never generate a GO here.
    """

    def __init__(self, output, study, *, resume, atomic_json, profile_points=None):
        self.output, self.atomic, self.study = Path(output), atomic_json, study
        self.path = self.output / "wave_state.json"
        self.study_sha = digest(study)
        self.profile_points = profile_points
        if profile_points is not None and (not isinstance(profile_points, list) or not profile_points):
            raise ValueError("profile_points must be a nonempty list, or omitted")
        if resume:
            self.state = json.loads(self.path.read_text())
            if (self.state.get("schema") != "gpuwrf.optimize.wave-state.v1"
                    or self.state.get("study_sha256") != self.study_sha):
                raise ValueError("wave checkpoint belongs to another study")
            next_wave, approvals, pending = (self.state["next_wave"], self.state["approvals"], self.state["pending"])
            if (type(next_wave) is not int or next_wave < 0 or not isinstance(approvals, list)
                    or len(approvals) != next_wave - (pending is not None)
                    or (pending is not None and pending["wave"] != next_wave-1)
                    or digest(self.state["checkpoint"]) != self.state["checkpoint_sha256"]):
                raise ValueError("invalid wave checkpoint/cursor or missing approval")
            self._approve()
        else:
            self.state = {"schema": "gpuwrf.optimize.wave-state.v1", "run_id": uuid.uuid4().hex,
                          "study_sha256": self.study_sha, "next_wave": 0, "pending": None,
                          "checkpoint": None, "checkpoint_sha256": digest(None), "approvals": []}
            self.atomic(self.path, self.state)
            self.atomic(self.output / "best.json", {"schema": "gpuwrf.optimize.best.v1",
                        "status": "NO_ADMISSIBLE_SOLUTION", "candidate": None})

    @property
    def checkpoint(self):
        return self.state["checkpoint"]

    def _approve(self):
        pending = self.state["pending"]
        if pending is None:
            return  # interrupted inside an already authorized wave
        report_path = self.output / pending["report"]
        if digest(json.loads(report_path.read_text())) != pending["report_sha256"]:
            raise ValueError("wave report changed since the checkpoint")
        if digest(self.checkpoint) != pending["checkpoint_sha256"]:
            raise ValueError("wave checkpoint changed since the report")
        path = self.output / pending["go_file"]
        if path.is_symlink():
            raise ValueError("wave GO must be a regular file, not a symlink")
        if not path.is_file() or path.stat().st_size > 16384:
            raise ValueError(f"external wave GO missing or oversized: {path}")
        token = json.loads(path.read_text())
        expected = {"schema": "gpuwrf.optimize.wave-go.v1",
                    "run_id": self.state["run_id"], "study_sha256": self.study_sha,
                    "wave": pending["wave"], "report_sha256": pending["report_sha256"],
                    "checkpoint_sha256": pending["checkpoint_sha256"]}
        required = set(expected) | {"approved_by", "decision"}
        if (not isinstance(token, dict) or not required <= set(token)
                or not set(token) <= required | {"assessment_sha256"}
                or token["decision"] not in ("GO", "STOP")
                or any(type(token.get(k)) is not type(v) or token[k] != v for k, v in expected.items())
                or not isinstance(token.get("approved_by"), str)
                or not re.fullmatch(r"alisios_mgr|review:[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", token["approved_by"])):
            raise ValueError("invalid or foreign wave GO; bind the exact report and checkpoint")
        if "assessment_sha256" in token:
            self._assessment(pending, token["assessment_sha256"])
        self.state["approvals"].append(token)
        self.state["pending"] = None
        self.state["stopped"] = token["decision"] == "STOP"
        self.atomic(self.path, self.state)  # consume BEFORE the next model launch

    def _assessment(self, pending, expected_sha):
        """Persist only the scoring authority's admissible winners; never infer admission from loss."""
        path = self.output / f"waves/wave-{pending['wave']:04d}.review.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8*1024**2:
            raise ValueError("wave assessment must be a small regular review file")
        review = json.loads(path.read_text())
        if (not isinstance(review, dict) or digest(review) != expected_sha
                or review.get("schema") != "gpuwrf.optimize.wave-assessment.v1"
                or review.get("run_id") != self.state["run_id"]
                or review.get("wave") != pending["wave"] or type(review.get("wave")) is not int
                or review.get("report_sha256") != pending["report_sha256"]):
            raise ValueError("invalid or foreign wave assessment")
        result = review.get("assessment", {})
        if (not isinstance(result, dict) or result.get("schema") != "gpuwrf.optimize.multiobjective.v1"
                or result.get("status") not in ("EVALUATED", "UNAVAILABLE")
                or not isinstance(result.get("scores", []), list)):
            raise ValueError("assessment requires multiobjective scores")
        objective = result.get("objective_identity")
        if result["status"] == "EVALUATED" and (not isinstance(objective, str)
                                                 or not re.fullmatch(r"[a-f0-9]{64}", objective)):
            raise ValueError("assessment requires the authority's frozen objective_identity")
        if objective is not None and self.state.get("objective_sha256") not in (None, objective):
            raise ValueError("assessment objective/denominator changed between waves")
        report = json.loads((self.output / pending["report"]).read_text())
        ensemble = {e if isinstance(e, str) else e["id"]: e for e in report["ensemble"]}
        eligible, seen = [], set()
        for row in result.get("scores", []):
            if not isinstance(row, dict) or row.get("id") not in ensemble or type(row.get("admissible")) is not bool:
                raise ValueError("assessment candidate or admissibility missing/foreign")
            if row["id"] in seen:
                raise ValueError("assessment repeats a candidate")
            seen.add(row["id"])
            if row["admissible"]:
                if result.get("status") in ("UNAVAILABLE", "MISSING_TRUTH", "FAILED"):
                    raise ValueError("unavailable truth cannot admit a best solution")
                if row.get("status") != "ADMISSIBLE":
                    raise ValueError("assessment admissibility contradicts its harm/support status")
                loss = row.get("loss")
                if type(loss) not in (int, float) or not math.isfinite(loss):
                    raise ValueError("admissible assessment requires a finite scalar loss")
                eligible.append(row)
        best_path = self.output / "best.json"
        best = json.loads(best_path.read_text())
        if objective is not None and best.get("objective_sha256") not in (None, objective):
            raise ValueError("best solution objective/denominator changed between waves")
        if eligible:
            row = min(eligible, key=lambda r: (r["loss"], r["id"]))
            if best["candidate"] is None or row["loss"] < best["candidate"]["loss"]:
                ledger_path = self.output / "ledger.json"
                runtime = (json.loads(ledger_path.read_text()).get("candidates", {}).get(row["id"])
                           if ledger_path.is_file() else None)
                best = {"schema": "gpuwrf.optimize.best.v1", "status": "ADMISSIBLE",
                        "candidate": row, "ensemble_member": ensemble[row["id"]],
                        "runtime": runtime,
                        "run_id": self.state["run_id"], "wave": pending["wave"],
                        "objective_sha256": objective, "report_sha256": pending["report_sha256"],
                        "assessment_sha256": expected_sha, "assessment_file": str(path)}
                self.atomic(best_path, best)
        self.state["last_assessment"] = {"path": str(path), "sha256": expected_sha,
                                         "objective_sha256": objective}
        if objective is not None:
            self.state["objective_sha256"] = objective

    def complete(self, number, *, kind, ensemble, scores, checkpoint, ledger,
                 impact_matrix=None, admissibility=None, accounting=None):
        from .study import Pause

        if number < self.state["next_wave"]:
            return  # deterministic discrete-rung replay, no new members dispatched
        if number != self.state["next_wave"]:
            raise ValueError("wave cursor skipped a boundary")
        members = {k: {name: v.get(name) for name in
                          ("rc", "wall_s", "vmhwm_kb", "vram_peak_mib", "aot_cache_warm")}
                   for k, v in sorted(ledger["members"].items())}
        report = {"schema": "gpuwrf.optimize.wave-report.v1", "run_id": self.state["run_id"],
                  "study_sha256": self.study_sha, "wave": number, "kind": kind,
                  "checkpoint_sha256": digest(checkpoint), "ensemble": ensemble, "scores": scores,
                  "question": {"region": self.study.get("region"), "objective": self.study.get("objective",
                               self.study.get("scorer")), "qualification": "external ALISIOS gates, diagnostics only"},
                  "effective_parameters": self.study.get("parameters", []), "inputs": self.study.get("inputs", []),
                  "control": self.study.get("configuration", {"kind": "external control required"}),
                  "impact_matrix": impact_matrix or {"status": "NOT_EVALUATED", "rows": []},
                  "admissibility": admissibility or {"status": "NOT_EVALUATED",
                      "reason": "external reference/control comparison required; no skill admission claimed"},
                  "resources": {"scheduled_case_hours": ledger["scheduled_case_hours"],
                                "completed_members": len(members), "members": members,
                                "accounting": accounting or {},
                                "failures": [b for b in ledger.get("batches", []) if b.get("rc") not in (None, 0)]},
                  "uncertainty": {"status": "NOT_EVALUATED"},
                  "balance_residuals": {"status": "UNAVAILABLE"},
                  "profiles": self._profiles(ensemble, ledger),
                  "ledger": "ledger.json",
                  "status": "AWAITING_EXTERNAL_GO"}
        root = self.output / "waves"
        root.mkdir(exist_ok=True)
        name = f"waves/wave-{number:04d}.json"
        self.atomic(self.output / name, report)
        self.state.update(next_wave=number+1, checkpoint=checkpoint, checkpoint_sha256=digest(checkpoint),
                          pending={"wave": number, "report": name, "report_sha256": digest(report),
                                   "checkpoint_sha256": digest(checkpoint),
                                   "go_file": f"waves/wave-{number:04d}.GO.json"})
        self.atomic(self.path, self.state)
        raise Pause(f"wave {number} complete; external GO required at {self.state['pending']['go_file']}")

    def _profiles(self, ensemble, ledger):
        if self.profile_points is None:
            return {"status": "UNAVAILABLE", "reason": "no profile points requested"}
        try:
            from .diagnostics import write_profile_report, wave_report_entry
        except ImportError as exc:
            raise ValueError("profile diagnostics support is unavailable in this build") from exc
        ids = {e if isinstance(e, str) else e["id"] for e in ensemble}
        folder = self.output / "diagnostics"
        folder.mkdir(exist_ok=True)
        entries = {}
        for key, member in sorted(ledger["members"].items()):
            if key.split("|", 1)[0] not in ids:
                continue
            files = member.get("files", [])
            if not files:
                entries[key] = {"status": "UNAVAILABLE", "reason": "member has no held histories"}
                continue
            paths = sorted(p for p in Path(files[0]["path"]).parent.glob("wrfout_d*") if not p.name.endswith(".tmp"))
            output = folder / (key.replace("|", "__") + ".profiles.json")
            if not output.exists():
                write_profile_report(paths, self.profile_points, output)
            entries[key] = wave_report_entry(output)
        return {"status": "RECORDED" if entries else "UNAVAILABLE", "members": entries, "points": self.profile_points,
                "qualification": "offline held histories; no model diagnostics/taps enabled"}
