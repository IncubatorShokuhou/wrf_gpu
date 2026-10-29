"""Paired, discrete input contrasts with opt-in host-loader consumer receipts.

Prepared cases belong to the caller (WPS/source adapters are outside the engine).
The control is executed unchanged. No hooks are installed in an ordinary run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SCHEMA = "gpuwrf.optimize.contrasts.v1"
PLAN = "gpuwrf.optimize.contrast-plan.v1"
RECEIPT = "gpuwrf.optimize.consumed-inputs.v1"
KINDS = ("warm-start", "ic", "lbc", "sst-held")
ATMOSPHERE = frozenset("U V W T THM P PH MU QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP QNICE QNRAIN QKE".split())
IC_REQUIRED = frozenset(("U", "V", "W", "T", "PH", "MU", "QVAPOR", "SST", "TSK"))
# History-only additions may accompany transported atmosphere. They are not
# physical interventions and never establish consumption of changed state.
WARM_DIAGNOSTICS = frozenset(("U10", "V10"))


def _identity(path):
    from .data import identity
    from gpuwrf.assimilation.letkf.alisios_guard import is_alisios_data
    if is_alisios_data(path):
        raise ValueError("contrast inputs must be private ALISIOS facade copies, never ALISIOS roots")
    return identity(path)


def _hash(value, *, strict=True):
    """Hash numeric readbacks independently of loader's lossless f32 -> f64 cast."""
    import numpy as np
    raw = np.ma.asarray(value)
    if strict and np.ma.getmaskarray(raw).any():
        raise ValueError("consumer field contains missing values")
    a = np.asarray(raw)
    if a.dtype.kind in "fiu":
        if strict and not np.isfinite(a).all():
            raise ValueError("consumer field contains nonfinite values")
        a = a.astype("<f8")
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def _inventory(path):
    import numpy as np
    from gpuwrf.io.netcdf_lock import Dataset
    before = _identity(path)
    with Dataset(path) as ds:
        fields = {}
        for name, var in ds.variables.items():
            timed = bool(var.dimensions and var.dimensions[0] == "Time")
            # One host field/record at a time, never the whole model or time series.
            records = ([_hash(var[i], strict=False) for i in range(var.shape[0])] if timed
                       else [_hash(var[:], strict=False)])
            fields[name] = {"dimensions": list(var.dimensions), "dtype": str(var.dtype), "records": records,
                            "attributes": {k: str(var.getncattr(k)) for k in var.ncattrs()}}
        result = {"identity": before, "dimensions": {k: len(v) for k, v in ds.dimensions.items()},
                  "attributes": {k: json.loads(json.dumps(ds.getncattr(k), default=lambda a: np.asarray(a).tolist()))
                                 for k in ds.ncattrs()}, "fields": fields}
    if _identity(path) != before:
        raise ValueError("input changed while inventorying it")
    return result


def _value(config, group, key, default=None):
    value = config.get(group, {}).get(key, default)
    return value[0] if isinstance(value, (list, tuple)) else value


def _case(item, base, hours):
    from gpuwrf.ens.workflows import _case as native_case
    name = item["id"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("contrast IDs must be safe unique names")
    directory = (base/item["input_dir"]).resolve(strict=True)
    _identity(directory/"namelist.input")  # Refuse ALISIOS roots before any native read.
    from gpuwrf.assimilation.letkf.alisios_guard import is_alisios_data
    for p in directory.iterdir():
        if p.name.startswith(("wrfinput_", "wrfbdy_", "wrflowinp_", "wrffdda_", "wrfndi_", "wrfsst_", "wrfchem_")) or p.suffix == ".TBL":
            if is_alisios_data(p):
                raise ValueError("contrast inputs must be private ALISIOS facade copies, never ALISIOS roots")
    native = native_case(directory, directory/"namelist.input", hours)
    count = _value(native["config"], "domains", "max_dom", 1)
    # All domains use the same aux4 lifecycle, including the held-SST arm.
    sst = native["config"].get("physics", {}).get("sst_update", 0)
    values = sst if isinstance(sst, (list, tuple)) else [sst]*count
    if len(values) not in (1, count) or any(v != 1 for v in values):
        raise ValueError("contrasts require sst_update=1 on every domain; held SST uses flat SST records")
    if _value(native["config"], "time_control", "restart", False):
        raise ValueError("contrast cases use cold/cycling initialization, not checkpoint resume")
    required = {f"wrfinput_d{i:02d}" for i in range(1, count+1)} | {"wrfbdy_d01"}
    required |= {f"wrflowinp_d{i:02d}" for i in range(1, count+1)}
    paths = {p.name: p for p in native["inputs"]}
    if not required <= paths.keys():
        raise ValueError("contrast cases require standard wrfinput/wrfbdy/wrflowinp names for every domain")
    files = {}
    for filename, path in sorted(paths.items()):
        files[filename] = (_inventory(path) if filename in required else {"identity": _identity(path)})
    for i in range(1, count+1):
        fields = files[f"wrfinput_d{i:02d}"]["fields"]
        if not IC_REQUIRED <= fields.keys():
            raise ValueError("IC consumer proof requires U/V/T/QVAPOR/SST/TSK")
        aux = files[f"wrflowinp_d{i:02d}"]["fields"]
        if not {"SST", "SEAICE", "VEGFRA", "ALBBCK", "Times"} <= aux.keys() or len(aux["SST"]["records"]) < 2:
            raise ValueError("SST contrast requires a complete multi-record aux4 series")
    return {"id": name, "kind": item.get("kind", "control"), "input_dir": str(directory),
            "namelist": _identity(directory/"namelist.input"), "config": native["config"],
            "issue": native["issue"].isoformat(), "max_dom": count, "files": files}


def _differences(control, arm):
    if arm["issue"] != control["issue"] or arm["max_dom"] != control["max_dom"]:
        raise ValueError("paired arms must have the same issue and domains")
    cfg, baseline = copy.deepcopy(arm["config"]), copy.deepcopy(control["config"])
    if arm["kind"] == "warm-start":
        if _value(cfg, "time_control", "cycling", False) is not True:
            raise ValueError("warm-start requires cycling=.true.")
        cfg["time_control"].pop("cycling", None)
        baseline["time_control"].pop("cycling", None)
    if cfg != baseline:
        raise ValueError("undeclared namelist change in paired arm")
    if arm["files"].keys() != control["files"].keys():
        raise ValueError("paired arms must contain the same input files")
    changes = []
    for filename, original in control["files"].items():
        changed = arm["files"][filename]
        if original["identity"]["sha256"] == changed["identity"]["sha256"]:
            continue
        warm_input = arm["kind"] == "warm-start" and filename.startswith("wrfinput_")
        same_dimensions = (original["dimensions"] == changed.get("dimensions") if not warm_input else
                           all(changed["dimensions"].get(k) == v for k, v in original["dimensions"].items()))
        if "fields" not in original or not same_dimensions:
            raise ValueError(f"undeclared table/geometry change: {filename}")
        allowed_attrs = {"history"}
        if arm["kind"] == "warm-start":
            allowed_attrs |= {"GPUWRF_ATMOSPHERE_CYCLE", "GPUWRF_LAND_CYCLE", "SAVE_TOPO_FROM_REAL", "TITLE"}
        if ({k: v for k, v in original["attributes"].items() if k not in allowed_attrs}
                != {k: v for k, v in changed["attributes"].items() if k not in allowed_attrs}):
            raise ValueError("undeclared input metadata change")
        a, b = original["fields"], changed["fields"]
        if arm["kind"] == "warm-start" and filename.startswith("wrfinput_"):
            # Only admitted atmospheric transport can change physical inputs.
            # Retain known history-only additions without inventing a transport claim.
            allowed = ATMOSPHERE | (WARM_DIAGNOSTICS - a.keys())
        elif arm["kind"] == "ic" and filename.startswith("wrfinput_"):
            allowed = ATMOSPHERE
        elif arm["kind"] in ("lbc", "warm-start") and filename == "wrfbdy_d01":
            allowed = set(a)
        elif arm["kind"] == "sst-held" and filename.startswith("wrflowinp_"):
            allowed = {"SST"}
        else:
            raise ValueError(f"undeclared input change: {filename}")
        for field in sorted(a.keys() | b.keys()):
            if a.get(field) == b.get(field):
                continue
            if field not in allowed or field == "Times":
                raise ValueError(f"undeclared field change: {filename}:{field}")
            if field in a and field in b and any(a[field][k] != b[field][k] for k in ("dimensions", "dtype", "attributes")):
                raise ValueError("paired field schema changed")
            if arm["kind"] == "warm-start" and filename == "wrfbdy_d01" and a[field]["records"][1:] != b[field]["records"][1:]:
                raise ValueError("cycling may update only the first LBC record; continued LBC must stay unchanged")
            if arm["kind"] == "sst-held":
                if any(row != a[field]["records"][0] for row in b[field]["records"]):
                    raise ValueError("held SST must repeat control record zero exactly")
            change = {"file": filename, "field": field}
            if warm_input and field in WARM_DIAGNOSTICS and field not in a:
                change["role"] = "retained-diagnostic"
            changes.append(change)
    if not changes:
        raise ValueError("contrast has no changed fields")
    return changes


def load_plan(path):
    path = Path(path).resolve(strict=True)
    _identity(path)
    raw = json.loads(path.read_text())
    if raw.get("schema") != SCHEMA or set(raw) - {"schema", "hours", "control", "arms", "member_timeout_s"}:
        raise ValueError(f"requires schema {SCHEMA} with only declared keys")
    if set(raw["control"]) != {"id", "input_dir"}:
        raise ValueError("control requires only id and input_dir")
    hours, cap = raw.get("hours"), raw.get("member_timeout_s", 1800)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (hours, cap)) or cap > 1800:
        raise ValueError("hours and member_timeout_s must be positive integers; member timeout <=1800s")
    control = _case(raw["control"], path.parent, hours)
    if control["kind"] != "control" or _value(control["config"], "time_control", "cycling", False):
        raise ValueError("control must be the unchanged cold-start case")
    arms = []
    for item in raw["arms"]:
        if item.get("kind") not in KINDS or set(item) != {"id", "kind", "input_dir"}:
            raise ValueError(f"arm requires id, input_dir and kind in {KINDS}")
        arm = _case(item, path.parent, hours)
        arm["changes"] = _differences(control, arm)
        if arm["kind"] == "warm-start":
            from gpuwrf.io.cycling_init import admit_cycling_input
            for i in range(1, arm["max_dom"]+1):
                filename = f"wrfinput_d{i:02d}"
                provenance = admit_cycling_input(Path(arm["input_dir"])/filename)
                if provenance.get("wrfinput_sha256") != control["files"][filename]["identity"]["sha256"]:
                    raise ValueError("warm-start provenance must bind the unchanged control analysis template")
                for change in arm["changes"]:
                    if (change["file"] == filename and change["field"] in ATMOSPHERE
                            and change["field"] not in provenance["fields"]):
                        raise ValueError("changed warm state must have admitted transport provenance")
        arms.append(arm)
    members = [control, *arms]
    if not arms or len({m["id"] for m in members}) != len(members):
        raise ValueError("need at least one contrast and unique member IDs")
    return {"schema": PLAN, "study": _identity(path), "hours": hours, "member_timeout_s": cap,
            "members": members, "planned_case_hours": hours*len(members),
            "status": "PLANNED; consumer proofs required before scoring"}


def _check_inputs(member):
    for record in [member["namelist"], *[v["identity"] for v in member["files"].values()]]:
        if _identity(record["path"]) != record:
            raise ValueError("declared input changed after planning")


class Consumer:
    """Record host arrays returned by the real loaders, without device reads."""
    def __init__(self, member):
        self.member, self.reads = member, {}
        self.paths = {str(Path(v["identity"]["path"]).resolve()): name for name, v in member["files"].items()}

    def observe(self, path, field, index, value, loader):
        filename = self.paths.get(str(Path(path).resolve()))
        if filename is None:
            raise ValueError(f"engine consumed an undeclared input: {Path(path).name}")
        spec = self.member["files"][filename].get("fields", {}).get(field)
        if spec is None:
            raise ValueError("engine consumed an undeclared field")
        actual = _hash(value)
        row = 0 if index is None else index
        if row < 0 or row >= len(spec["records"]) or actual != spec["records"][row]:
            raise ValueError(f"consumer readback differs: {filename}:{field}[{row}]")
        self.reads[(filename, field, row)] = {"file": filename, "field": field, "record": row,
                                              "loader": loader, **actual}

    def finish(self):
        required = set()
        retained = []
        for filename, file in self.member["files"].items():
            if filename.startswith("wrfinput_"):
                fields = set(IC_REQUIRED)
                for change in self.member.get("changes", []):
                    if change["file"] != filename:
                        continue
                    field = change["field"]
                    if self.member["kind"] == "warm-start":
                        if field in WARM_DIAGNOSTICS and change.get("role") == "retained-diagnostic":
                            retained.append({"file": filename, "field": field, "status": "RETAINED_ONLY_NOT_CONSUMER_PROOF"})
                            continue
                        if field not in ATMOSPHERE:
                            raise ValueError(f"unsupported physical warm-input change: {filename}:{field}")
                    fields.add(field)
                required |= {(filename, field, 0) for field in fields}
            elif filename.startswith("wrflowinp_"):
                required |= {(filename, field, i) for field in ("SST", "SEAICE", "VEGFRA", "ALBBCK")
                             for i in range(len(file["fields"][field]["records"]))}
            elif filename == "wrfbdy_d01":
                changed = {c["field"] for c in self.member.get("changes", []) if c["file"] == filename}
                required |= {(filename, field, i) for field, spec in file["fields"].items()
                             if (field.partition("_")[0] in {"U", "V", "W", "T", "PH", "MU", "QVAPOR"}
                                 and "_B" in field) or field in changed
                             for i in range(len(spec["records"]))}
        missing = required - self.reads.keys()
        if missing:
            raise ValueError(f"declared inputs were not consumed: {sorted(missing)[:8]}")
        _check_inputs(self.member)
        return {"schema": RECEIPT, "member": self.member["id"], "kind": self.member["kind"],
                "inputs": {k: v["identity"] for k, v in self.member["files"].items()},
                "namelist": self.member["namelist"], "reads": [self.reads[k] for k in sorted(self.reads)],
                "retained_diagnostics": retained,
                "status": "CONSUMED", "scope": "host loader ingress; no per-step application or fidelity claim"}


@contextmanager
def consumer_reads(consumer):
    """Install only around one isolated member. Restore all functions on failure."""
    from unittest.mock import patch
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io import boundary_replay, lower_boundary
    read_ic = Gen2Run._read_variable
    read_bdy = boundary_replay._read_wrfbdy_array
    dataset = lower_boundary.Dataset

    def ic(run, path, field, index):
        value = read_ic(run, path, field, index)
        consumer.observe(path, field, index, value, "Gen2Run._read_variable")
        return value

    def bdy(ds, field, index):
        value = read_bdy(ds, field, index)
        consumer.observe(ds.filepath(), field, index, value, "decode_wrfbdy")
        return value

    class Variable:
        def __init__(self, variable, path, field):
            self.variable, self.path, self.field = variable, path, field

        def __getitem__(self, key):
            value = self.variable[key]
            if self.field != "Times":
                # load_lower_boundary reads all records once, before jnp.asarray.
                for i, row in enumerate(value):
                    consumer.observe(self.path, self.field, i, row, "load_lower_boundary")
            return value

    class Aux:
        def __init__(self, ds, path):
            self.ds, self.path = ds, path

        def __getitem__(self, field):
            return Variable(self.ds[field], self.path, field)

    @contextmanager
    def aux(path, *args, **kwargs):
        with dataset(path, *args, **kwargs) as ds:
            yield Aux(ds, path)

    with patch.object(Gen2Run, "_read_variable", ic), \
         patch.object(boundary_replay, "_read_wrfbdy_array", bdy), \
         patch.object(lower_boundary, "Dataset", aux):
        yield


def validate_receipt(member, receipt):
    if (receipt.get("schema") != RECEIPT or receipt.get("status") != "CONSUMED"
            or receipt.get("member") != member["id"] or receipt.get("kind") != member["kind"]
            or receipt.get("inputs") != {k: v["identity"] for k, v in member["files"].items()}
            or receipt.get("namelist") != member["namelist"]):
        raise ValueError("foreign/incomplete consumer receipt")
    check = Consumer(member)
    for row in receipt["reads"]:
        filename, field, i = row["file"], row["field"], row["record"]
        spec = member["files"][filename]["fields"][field]
        if {k: row[k] for k in ("shape", "sha256")} != spec["records"][i] or i < 0:
            raise ValueError("consumer receipt field readback differs")
        expected = ("Gen2Run._read_variable" if filename.startswith("wrfinput_") else
                    "decode_wrfbdy" if filename == "wrfbdy_d01" else "load_lower_boundary")
        if row["loader"] != expected:
            raise ValueError("unknown consumer loader")
        check.reads[(filename, field, i)] = row
    expected = check.finish()
    if receipt.get("retained_diagnostics", []) != expected["retained_diagnostics"]:
        raise ValueError("consumer receipt retained-diagnostic disclosure differs")


def member_run(plan, member, output, *, model_main=None):
    from .data import write_json
    _check_inputs(member)
    output = Path(output)
    if output.exists():
        raise ValueError("member output exists; refusing overwrite")
    consumer = Consumer(member)
    native = model_main is None
    if model_main is None:
        if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu" and os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
            raise ValueError("GPU contrast member requires official GPU lock")
        from gpuwrf.cli import main as model_main
    started = time.monotonic()
    with consumer_reads(consumer):
        rc = model_main(["run", "--input-dir", member["input_dir"], "--domains-from-namelist",
                         "--hours", str(plan["hours"]), "--output-dir", str(output/"forecast")])
    if rc != 0:
        raise ValueError(f"contrast member failed with exit {rc}; no consumer receipt")
    receipt = consumer.finish()
    platform = "stub; no forecast qualification"
    if native:
        import jax
        platform = jax.devices()[0].platform
        if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu" and platform != "gpu":
            raise ValueError("GPU contrast member ran on a different platform")
    receipt.update(hours=plan["hours"], platform=platform, wall_s=time.monotonic()-started)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"consumer.json", receipt)
    return receipt


def _lock_fds(env):
    """Forward one official lock capability, never all inherited descriptors.

    The child's existing token/holder/descriptor preflight remains authoritative.
    CPU execution without a lock needs no descriptor or lock-file access.
    """
    from gpuwrf.ens.workflows import GPU_LOCK_FILE
    text = env.get("GPUWRF_GPU_LOCK_FD", "")
    if not text and env.get("JAX_PLATFORMS", "").strip().lower() == "cpu" and env.get("GPUWRF_GPU_LOCK_HELD") != "1":
        return ()
    if (not isinstance(text, str) or not text.isascii() or not text.isdecimal()
            or int(text) <= 2 or env.get("GPUWRF_GPU_LOCK_HELD") != "1"):
        raise ValueError("invalid GPU lock descriptor")
    descriptor = int(text)
    try:
        held, official = os.fstat(descriptor), GPU_LOCK_FILE.stat()
    except OSError as exc:
        raise ValueError("GPU lock descriptor is unavailable") from exc
    if (held.st_dev, held.st_ino) != (official.st_dev, official.st_ino):
        raise ValueError("GPU lock descriptor does not name the official lock")
    return (descriptor,)


def run(plan, output, *, member_env=None, runner=None):
    """Sequential bounded members; external wave/registry authority wraps this wave.

    A runner callback is for CPU orchestration tests. The default launches a fresh
    member process, so the CPU optimizer never owns a CUDA pool.
    """
    from .data import write_json
    env = dict(os.environ if member_env is None else member_env)
    if env.get("JAX_PLATFORMS", "").lower() != "cpu" and env.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise ValueError("GPU contrast wave requires scripts/with_gpu_lock.sh")
    pass_fds = _lock_fds(env)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output/"plan.json"
    write_json(plan_path, plan)
    # Set allocator before interpreter startup; prevent CLI re-exec dropping the observer.
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    results = []
    for member in plan["members"]:
        _check_inputs(member)
        destination = output/member["id"]
        if runner is None:
            subprocess.run([sys.executable, "-m", __name__, "member", "--plan", str(plan_path),
                            "--arm", member["id"], "--output", str(destination)],
                           check=True, timeout=plan["member_timeout_s"], env=env, pass_fds=pass_fds)
        else:
            runner(plan, member, destination)
        receipt_path = destination/"consumer.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("hours") != plan["hours"]:
            raise ValueError("consumer receipt horizon differs")
        validate_receipt(member, receipt)
        results.append({"id": member["id"], "kind": member["kind"], "receipt": _identity(receipt_path),
                        "forecast_dir": str(destination/"forecast"), "changes": member.get("changes", [])})
    report = {"schema": "gpuwrf.optimize.contrast-result.v1", "status": "CONSUMED; unscored paired arms",
              "control": plan["members"][0]["id"], "members": results,
              "planned_case_hours": plan["planned_case_hours"]}
    write_json(output/"result.json", report)
    return report


def register_parser(actions, command):
    parser = actions.add_parser("contrasts", help="paired IC/LBC, warm-start and held-SST input studies")
    sub = parser.add_subparsers(dest="contrast_action", required=True)
    for action in ("plan", "run"):
        child = sub.add_parser(action)
        child.add_argument("--study", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.set_defaults(func=command)


def command(args, member_env):
    from .data import write_json
    plan = load_plan(args.study)
    if args.contrast_action == "plan":
        write_json(args.output, plan)
        result = plan
    else:
        try:
            result = run(plan, args.output, member_env=member_env)
        except subprocess.SubprocessError as exc:
            raise ValueError(f"contrast member process failed: {type(exc).__name__}") from exc
    print(json.dumps({"status": result["status"], "planned_case_hours": result["planned_case_hours"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("member",))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN:
        raise ValueError("unsupported contrast plan")
    member = next(m for m in plan["members"] if m["id"] == args.arm)
    member_run(plan, member, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
