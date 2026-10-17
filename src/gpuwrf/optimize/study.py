"""Admitted physics search, date-separated successive halving, ensemble launch."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import pandas as pd

from .data import OBJECTIVES, identity, metrics, paired, read_csv, uncertainty, write_json

PHYSICS = ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
           "ra_lw_physics", "ra_sw_physics", "cu_physics")


def model_digest():
    """Source binding for operator admissions, excluding offline tools/CLI."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [root/name for name in ("__init__.py", "_fast_defaults.py", "_x64_config.py")]
    for directory in ("contracts", "coupling", "dynamics", "physics", "kernels", "runtime", "integration", "nesting", "io", "dfi"):
        paths.extend(p for p in (root/directory).rglob("*") if p.is_file() and
                     p.suffix.lower() in (".py", ".npz", ".json", ".csv", ".tbl"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes()+b"\0")
    return digest.hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _physics(raw, max_dom):
    if set(raw) != set(PHYSICS):
        raise ValueError(f"every admission must specify exactly {PHYSICS}")
    result = {}
    for key, value in raw.items():
        values = value if isinstance(value, list) else [value]*max_dom
        if len(values) != max_dom or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError("physics codes must be nonnegative integers for every domain")
        result[key] = values
    return result


def config_id(physics):
    return "p-"+hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()[:12]


def namelist_text(config):
    def scalar(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return ".true." if value else ".false."
        if isinstance(value, str):
            return "'"+value.replace("'", "''")+"'"
        return str(value)
    return "\n".join("&"+group+"\n"+"\n".join(
        " "+key+" = "+", ".join(scalar(v) for v in (value if isinstance(value, list) else [value]))+","
        for key, value in values.items())+"\n/" for group, values in config.items())+"\n"


def load_study(path):
    from gpuwrf import __version__
    from gpuwrf.io.namelist_check import (validate_operational_namelist, refuse_single_domain_mp_nested,
                                        refuse_unqualified_nested_options)
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.ens.workflows import _case

    path = Path(path).resolve(strict=True)
    raw = json.loads(path.read_text())
    if raw.get("schema") != "gpuwrf.optimize.study.v1" or not str(raw.get("region", "")).strip():
        raise ValueError("study requires schema gpuwrf.optimize.study.v1 and a region")
    if raw.get("objective") not in OBJECTIVES:
        raise ValueError(f"objective must be one of {tuple(OBJECTIVES)}")
    resolve = lambda p: (path.parent/str(p)).resolve(strict=True)
    variables = OBJECTIVES[raw["objective"]]
    scales = raw.get("scales", {v: 1.0 for v in variables})
    if set(scales) != set(variables) or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("scales must specify a positive finite physical-unit scale per objective variable")
    if raw["objective"] == "all" and "scales" not in raw:
        raise ValueError("all requires explicit scales (mixed units cannot be added directly)")
    hours = _positive_int(raw.get("hours"), "hours")
    budget = _positive_int(raw.get("budget_case_hours"), "budget_case_hours")
    split = pd.Timestamp(raw["test_start"], tz="UTC")
    if split != split.normalize():
        raise ValueError("test_start must be a UTC date")
    cases = []
    for item in raw["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", item["id"]):
            raise ValueError("case IDs must be safe unique names")
        date = pd.Timestamp(item["date"], tz="UTC")
        if date != date.normalize():
            raise ValueError("case date must be YYYY-MM-DD")
        directory = resolve(item["input_dir"])
        native = _case(directory, directory/"namelist.input", hours)
        config = native["config"]
        if native["issue"].strftime("%Y-%m-%d") != date.strftime("%Y-%m-%d"):
            raise ValueError("declared case date disagrees with namelist/wrfinput initialization")
        count = config.get("domains", {}).get("max_dom", 1)
        count = _positive_int(count[0] if isinstance(count, list) else count, "max_dom")
        domain = item.get("domain", f"d{count:02d}")
        if domain not in [f"d{n:02d}" for n in range(1, count+1)]:
            raise ValueError("case scoring domain does not exist")
        cases.append({"id": item["id"], "date": date.strftime("%Y-%m-%d"), "input_dir": str(directory),
                      "domain": domain, "config": config, "max_dom": count, "issue": native["issue"].isoformat(),
                      "source_namelist": identity(directory/"namelist.input"),
                      "split": "train" if date < split else "test"})
    if len({c["id"] for c in cases}) != len(cases) or len({c["date"] for c in cases}) != len(cases):
        raise ValueError("need unique case IDs and one initialization per UTC date")
    if len({c["input_dir"] for c in cases}) != len(cases):
        raise ValueError("duplicate input case directory")
    if len({c["max_dom"] for c in cases}) != 1:
        raise ValueError("study cases must have the same domain count")
    if any(sum(c["split"] == s for c in cases) < 2 for s in ("train", "test")):
        raise ValueError("need at least two train dates and two held-out test dates")
    max_dom = cases[0]["max_dom"]
    source = model_digest()
    admissions = {}
    evidence = []
    for receipt_path in raw["admissions"]:
        receipt_path = resolve(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "gpuwrf.optimize.admission.v1" or receipt.get("model_sha256") != source
                or receipt.get("gpuwrf_version") != __version__ or receipt.get("max_dom") != max_dom
                or receipt.get("status") not in ("GPU_QUALIFIED", "GPU_SMOKED_ADMITTED")):
            raise ValueError("admission must bind this model source, version, domain count and explicit GPU qualification")
        physics = _physics(receipt["physics"], max_dom)
        if not receipt.get("evidence"):
            raise ValueError("admission requires retained GPU evidence identities")
        for record in receipt["evidence"]:
            actual = identity(receipt_path.parent/record["path"])
            if actual["sha256"] != record["sha256"]:
                raise ValueError("GPU admission evidence changed")
            evidence.append(actual)
        key = config_id(physics)
        if key in admissions:
            raise ValueError("duplicate physics admission")
        admissions[key] = {"id": key, "physics": physics, "admission": identity(receipt_path), "status": receipt["status"]}
    space = raw.get("search_space")
    if space is not None:
        if set(space) != set(PHYSICS) or any(not isinstance(v, list) or not v for v in space.values()):
            raise ValueError("search_space requires a nonempty options list for each physics key")
        if math.prod(map(len, space.values())) > 4096:
            raise ValueError("search_space exceeds 4096 combinations; narrow it")
        requested = {config_id(_physics(dict(zip(PHYSICS, values)), max_dom))
                     for values in itertools.product(*(space[key] for key in PHYSICS))}
        admissions = {key: value for key, value in admissions.items() if key in requested}
    candidates, rejected = [], []
    for candidate in admissions.values():
        try:
            for case in cases:
                cfg = copy.deepcopy(case["config"])
                cfg.setdefault("physics", {}).update(candidate["physics"])
                validate_operational_namelist(cfg)
                refuse_single_domain_mp_nested(cfg, max_dom)
                refuse_unqualified_nested_options(cfg, max_dom)
                warnings = require_namelist_honoured(cfg, tuple(f"d{i:02d}" for i in range(1, max_dom+1)))
                candidate.setdefault("warnings", {})[case["id"]] = warnings
            candidates.append(candidate)
        except ValueError as exc:
            rejected.append({"id": candidate["id"], "reason": str(exc)})
    if not candidates:
        raise ValueError("no admitted WRF-valid configurations; CPU-only/catalog status is insufficient")
    stations, obs = resolve(raw["stations"]), resolve(raw["observations"])
    study = {"schema": raw["schema"], "region": raw["region"], "objective": raw["objective"], "variables": variables,
             "scales": scales, "hours": hours, "budget_case_hours": budget, "test_start": split.isoformat(),
             "cases": sorted(cases, key=lambda c: c["date"]), "candidates": sorted(candidates, key=lambda c: c["id"]),
             "stations": str(stations), "observations": str(obs), "sampling": raw.get("sampling", {}),
             "model_sha256": source, "inputs": [identity(p) for p in (path, stations, obs)]+evidence+
                 [c["source_namelist"] for c in cases],
             "rejected": rejected, "qualification": "operator admissions; existence/hash checked, scientific verdict not inferred"}
    study["rounds"], study["planned_case_hours"] = schedule(len(candidates), sum(c["split"] == "train" for c in cases),
        sum(c["split"] == "test" for c in cases), hours, budget)
    return study


def schedule(count, train_dates, test_dates, hours, budget):
    """Double training dates, halve candidates; reserve all test dates for winner."""
    rounds, used, previous, target = [], test_dates*hours, 0, min(3 if train_dates == 3 else 2, train_dates)
    while True:
        used += count*(target-previous)*hours
        rounds.append({"candidates": count, "train_dates": target})
        if target == train_dates:
            break
        count = max(1, math.ceil(count/2))
        previous, target = target, min(train_dates, target*2)
        if train_dates-target == 1:
            target = train_dates
    if used > budget:
        raise ValueError(f"budget needs at least {used} case-hours including held-out winner (given {budget})")
    return rounds, used


def evaluate(study, fetch):
    """fetch(candidate, cases) is lazy; losing configurations never read test truth."""
    from gpuwrf.verify.observations import KEYS, read_stations

    stations = read_stations(study["stations"])
    observations = read_csv(study["observations"], observations=True)
    train = [c for c in study["cases"] if c["split"] == "train"]
    test = [c for c in study["cases"] if c["split"] == "test"]
    cutoff = pd.Timestamp(study["test_start"])
    candidates = study["candidates"]
    cache, rounds, provenance = {}, [], list(study["inputs"])

    def collect(candidate, cases):
        missing = [c for c in cases if (candidate["id"], c["id"]) not in cache]
        results = fetch(candidate, missing) if missing else {}
        for case in missing:
            rows, audit, identities = results[case["id"]]
            part = paired(observations, rows, stations, study["variables"])
            start = pd.Timestamp(case["issue"])
            if rows.valid_time.lt(start).any() or rows.valid_time.gt(start+pd.Timedelta(hours=study["hours"])).any():
                raise ValueError("forecast support outside declared case date/horizon")
            part = part.loc[part.valid_time.gt(start)]  # initialization is not forecast skill
            # Train/test split applies to actual scored intervals as well as issue dates.
            part = part.loc[part.valid_time.lt(cutoff)] if case["split"] == "train" else part.loc[part.interval_start.ge(cutoff)]
            if part.empty:
                raise ValueError("no support after date split")
            part["case_id"] = case["id"]
            cache[candidate["id"], case["id"]] = part
            provenance.append({"candidate": candidate["id"], "case": case["id"], "extraction": audit, "inputs": identities})
        return pd.concat([cache[candidate["id"], c["id"]] for c in cases], ignore_index=True)

    for number, spec in enumerate(study["rounds"]):
        frames = {c["id"]: collect(c, train[:spec["train_dates"]]) for c in candidates}
        keys = KEYS+["case_id"]
        support = None
        for frame in frames.values():
            rows = frame[keys]
            support = rows if support is None else support.merge(rows, on=keys, validate="one_to_one")
        scores = []
        for candidate in candidates:
            frame = frames[candidate["id"]].merge(support, on=keys, validate="one_to_one")
            if set(frame.case_id) != {c["id"] for c in train[:spec["train_dates"]]}:
                raise ValueError("one or more dates have no common support across candidates")
            scores.append({"id": candidate["id"], **metrics(frame, study["variables"], study["scales"])})
        scores.sort(key=lambda s: (s["loss"], s["id"]))
        rounds.append({"round": number, "train_dates": spec["train_dates"], "scores": scores,
                       "common_rows": len(support)})
        survivors = {r["id"] for r in scores[:max(1, math.ceil(len(scores)/2))]}
        candidates = [c for c in candidates if c["id"] in survivors]
    winner = next(c for c in study["candidates"] if c["id"] == rounds[-1]["scores"][0]["id"])
    held = collect(winner, test)
    for _, date_rows in held.groupby(held.valid_time.dt.date):
        metrics(date_rows, study["variables"], study["scales"])
    report = {"schema": "gpuwrf.optimize.result.v1", "region": study["region"], "objective": study["objective"],
              "recommended": winner, "rounds": rounds,
              "held_out": {**metrics(held, study["variables"], study["scales"]),
                           "uncertainty": uncertainty(held, study["variables"], study["scales"])},
              "budget_case_hours": study["budget_case_hours"], "planned_case_hours": study["planned_case_hours"],
              "evaluated_case_hours": len(cache)*study["hours"], "test_start": study["test_start"],
              "train_cases": [c["id"] for c in train], "test_cases": [c["id"] for c in test],
              "model_sha256": study["model_sha256"], "provenance": provenance,
              "status": "HELD_OUT_DIAGNOSTIC; no release promotion or guarantee of regional superiority"}
    json.dumps(report, allow_nan=False)
    return report


def score(study, inventory_path):
    from .data import forecast
    from gpuwrf.verify.observations import read_stations

    inventory_path = Path(inventory_path).resolve(strict=True)
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema") != "gpuwrf.optimize.forecasts.v1":
        raise ValueError("unsupported forecast inventory")
    stations = read_stations(study["stations"])
    study = copy.deepcopy(study)
    study["inputs"].append(identity(inventory_path))
    def fetch(candidate, cases):
        results = {}
        for case in cases:
            files = [inventory_path.parent/p for p in inventory["forecasts"][candidate["id"]][case["id"]]]
            rows, audit = forecast(files, stations, **study["sampling"])
            results[case["id"]] = (rows, audit, [identity(p) for p in files])
        return results
    return evaluate(study, fetch)


def run(study, output, *, gpus=None, max_parallel=1, launcher=None):
    """No JAX initialization here; existing ensemble launcher owns GPU execution."""
    from gpuwrf.ens.workflows import prepare, launch
    from gpuwrf.verify.observations import read_stations
    from .data import forecast

    _positive_int(max_parallel, "max_parallel")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"study.json", study)
    stations = read_stations(study["stations"])
    batch_number = 0
    spent = 0
    def fetch(candidate, cases):
        nonlocal batch_number, spent
        if spent+len(cases)*study["hours"] > study["budget_case_hours"]:
            raise ValueError("compute budget exhausted before launch")
        if len(cases) < 2:
            raise ValueError("ensemble batch launcher requires at least two new cases per round")
        batch = output/f"batch-{batch_number:03d}"
        batch_number += 1
        inputs = []
        for case in cases:
            dest = output/"inputs"/candidate["id"]/case["id"]
            dest.mkdir(parents=True)
            cfg = copy.deepcopy(case["config"])
            cfg.setdefault("physics", {}).update(candidate["physics"])
            (dest/"namelist.input").write_text(namelist_text(cfg))
            for source in Path(case["input_dir"]).iterdir():
                if source.name != "namelist.input" and source.is_file() and (source.name.startswith(("wrfinput_", "wrfbdy_", "wrffdda_", "wrflowinp_")) or source.suffix == ".TBL"):
                    (dest/source.name).symlink_to(source.resolve())
            inputs.append(dest)
        plan = prepare("batch", batch, cases=inputs, hours=study["hours"], gpus=gpus, max_parallel=max_parallel)
        spent += len(cases)*study["hours"]
        write_json(batch/"optimizer_budget.json", {"scheduled_case_hours": spent, "budget_case_hours": study["budget_case_hours"]})
        rc = launch(batch/"plan.json", launcher=launcher)
        if rc:
            raise ValueError(f"ensemble batch failed with exit {rc}; no recommendation published")
        results = {}
        for case, member in zip(cases, plan["members"]):
            files = sorted((Path(plan["out_root"])/member["id"]/"wrfout").glob(f"wrfout_{case['domain']}_*"))
            if not files:
                raise ValueError("scheduled forecast produced no scoring-domain wrfouts")
            rows, audit = forecast(files, stations, **study["sampling"])
            results[case["id"]] = (rows, audit, [identity(p) for p in files]+[identity(batch/"plan.json")])
        return results
    report = evaluate(study, fetch)
    report["scheduled_case_hours"] = spent
    write_json(output/"result.json", report)
    return report
