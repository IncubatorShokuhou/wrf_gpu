"""Optimizer-to-scheduler registry metadata; backend-dark, no production reads."""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import uuid


def batch_metadata(study, plan, pairs, inputs, policy):
    from .study import _source_case

    by_source = {str(p.resolve()): case for (_, case), p in zip(pairs, inputs)}
    members = {}
    for member in plan["members"]:
        case = by_source[str(_source_case(member))]
        issue = datetime.fromisoformat(case["issue"]).strftime("%Y%m%d_%Hz")
        members[str(Path(member["case"]).resolve())] = {
            "issue": issue, "lead_h": study["hours"], "nests": f"{case['max_dom']} domains"}
    allowed = {"forcing", "atlas_dir", "smoke", "by", "producer"}
    if set(policy) - allowed:
        raise ValueError(f"unknown optimizer registry policy keys: {sorted(set(policy)-allowed)}")
    return {"schema": "gpuwrf.optimize.registry.v1", "members": members, "policy": policy,
            "purpose": f"Optimizer wave: {study['region']}; diagnostic, no promotion"}


def load_hooks():
    name = "_gpuwrf_optimize_alisios_registry"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[3] / "scripts" / "alisios_registry.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("optimizer registry requires scripts/alisios_registry.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Invocation:
    """One real member attempt, including retries; finish exactly once in scheduler finally."""

    def __init__(self, metadata, case, output, *, factory=None):
        data = json.loads(Path(metadata).read_text())
        if data.get("schema") != "gpuwrf.optimize.registry.v1":
            raise ValueError("invalid optimizer registry metadata")
        member = data["members"][str(Path(case).resolve())]
        policy = data["policy"]
        hooks = load_hooks()
        self.registry = (factory or hooks.Registry)(atlas_dir=policy.get("atlas_dir"), smoke=policy.get("smoke", False))
        self.run_id = "optimize:" + uuid.uuid4().hex
        self.start = {"run_id": self.run_id, "by": policy.get("by", "gpuwrf-optimize"),
                      "producer": policy.get("producer", "gpuwrf-optimize"), "family": "optimizer",
                      "purpose": data["purpose"], "forcing": policy["forcing"], "engine": "wrf_gpu",
                      "run_dir": str(Path(output).resolve()), "retention": "raw_keep", **member}
        self.context = None
        self.complete = False

    def begin(self, env):
        context = self.registry.run(self.start, end={"corpus": "held", "raw_deleted": False})
        self.outcome = context.__enter__()  # schema and start append before Popen
        self.context = context
        return dict(env, ALISIOS_WRF_RUN_ID=self.run_id, ALISIOS_ATLAS_DIR=str(self.registry.atlas))

    def finish(self, rc, *, aborted=False):
        context, self.context = self.context, None
        if context is None:
            return
        self.outcome.rc = rc if isinstance(rc, int) else 1
        if aborted or (isinstance(rc, int) and rc < 0):
            error = SystemExit(128-rc if isinstance(rc, int) and rc < 0 else (rc or 130))
            context.__exit__(type(error), error, None)
        else:
            context.__exit__(None, None, None)
        self.complete = True
