"""Released physics search, date-separated successive halving (rungs), resumable ensemble launch."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import time

import pandas as pd

from . import released
from .data import OBJECTIVES, identity, metrics, paired, read_csv, uncertainty, write_json
from .released import PHYSICS, config_id  # noqa: F401  (re-exported: stable optimize API)

LEDGER = "ledger.json"
PAUSED = 3  # CLI exit status: chunk budget used, resume with --resume


class Pause(Exception):
    """The chunk wall budget would be exceeded by the next launcher batch."""


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


def record_signature(directory, max_dom):
    """Time-record counts of the boundary/lower-boundary inputs (program shapes until w41-bdyrec)."""
    from gpuwrf.io.netcdf_lock import Dataset

    signature = {}
    for name in ["wrfbdy_d01"]+[f"wrflowinp_d{n:02d}" for n in range(1, max_dom+1)]:
        path = Path(directory)/name
        if path.is_file():
            with Dataset(path) as ds:
                signature[name] = len(ds.dimensions["Time"])
    return signature


FORCING_PREFIXES = ("wrfinput_", "wrfbdy_", "wrffdda_", "wrflowinp_")


def forecast_inputs(directory):
    """The case files a member consumes besides the namelist (linked into every member input directory)."""
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.name != "namelist.input"
                  and (p.name.startswith(FORCING_PREFIXES) or p.suffix == ".TBL"))


def mem_available_gb():
    """Host MemAvailable in GB, read like the launcher does (None when unreadable)."""
    try:
        with open("/proc/meminfo") as stream:
            return next(int(line.split()[1]) for line in stream if line.startswith("MemAvailable"))/1e6
    except (OSError, ValueError, StopIteration):
        return None


def _geometry(directory):
    from gpuwrf.io.netcdf_lock import Dataset

    with Dataset(Path(directory)/"wrfinput_d01") as ds:
        return {key: len(ds.dimensions[key]) for key in ("west_east", "south_north", "bottom_top")}


def _candidates(raw, data, max_dom, e41, receipts, resolve, source):
    """Operator admissions and released configurations; every one must be in the release list."""
    from gpuwrf import __version__

    admissions, evidence = {}, []
    for receipt_path in receipts:
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
        entry = released.resolve(data, physics, max_dom, e41=e41)  # an admission can never widen the release
        key = config_id(physics)
        if key in admissions:
            raise ValueError("duplicate physics admission")
        admissions[key] = {"id": key, "physics": physics, "admission": identity(receipt_path),
                           "status": receipt["status"], "entry": entry}
    names = raw.get("configurations", [])
    if names == "all":
        entries = [e for e in data["configurations"] if e["max_dom"] == max_dom and (e41 == "any" or e["best_E41"] == "PASS")]
    elif isinstance(names, list) and all(isinstance(n, str) for n in names):
        entries = []
        for name in names:
            entry = released.lookup(data, name)
            entries.append(released.resolve(data, entry["physics"], max_dom, e41=e41))
        if len({e["id"] for e in entries}) != len(entries):
            raise ValueError("configurations name the same released configuration twice")
    else:
        raise ValueError("configurations must be 'all' or a list of released IDs/run names")
    for entry in entries:
        admissions.setdefault(entry["id"], {"id": entry["id"], "physics": entry["physics"], "admission": None,
                                            "status": f"RELEASED_{data['release']}", "entry": entry})
    if not admissions:
        raise ValueError("study needs configurations (released IDs/run names or 'all') or admissions")
    return admissions, evidence


def load_study(path):
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
    e41 = raw.get("e41", "any")
    if e41 not in released.E41_LEVELS:
        raise ValueError(f"e41 must be one of {released.E41_LEVELS}")
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
                      # Bound by content: a resume refuses a study whose case inputs changed (one streaming hash each).
                      "forcing": [identity(p) for p in forecast_inputs(directory)],
                      "records": record_signature(directory, count), "geometry": _geometry(directory),
                      "split": "train" if date < split else "test"})
    if len({c["id"] for c in cases}) != len(cases) or len({c["date"] for c in cases}) != len(cases):
        raise ValueError("need unique case IDs and one initialization per UTC date")
    if len({c["input_dir"] for c in cases}) != len(cases):
        raise ValueError("duplicate input case directory")
    if len({c["max_dom"] for c in cases}) != 1:
        raise ValueError("study cases must have the same domain count")
    if len({json.dumps(c["geometry"], sort_keys=True) for c in cases}) != 1:
        raise ValueError("study cases must share one d01 geometry (one program per configuration)")
    if any(sum(c["split"] == s for c in cases) < 2 for s in ("train", "test")):
        raise ValueError("need at least two train dates and two held-out test dates")
    max_dom = cases[0]["max_dom"]
    source = model_digest()
    data = released.load()
    admissions, evidence = _candidates(raw, data, max_dom, e41, raw.get("admissions", []), resolve, source)
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
        entry = candidate.pop("entry")
        candidate["release"] = {"release": data["release"], "best_E41": entry["best_E41"],
                                "qualified_by": [{k: r[k] for k in ("run", "geography", "D6", "E41", "gate_hours")}
                                                 for r in entry["qualified_by"]]}
        try:
            for case in cases:
                cfg = copy.deepcopy(case["config"])
                cfg.setdefault("physics", {}).update(candidate["physics"])
                validate_operational_namelist(cfg)
                refuse_single_domain_mp_nested(cfg, max_dom)
                refuse_unqualified_nested_options(cfg, max_dom)
                warnings = require_namelist_honoured(cfg, tuple(f"d{i:02d}" for i in range(1, max_dom+1)))
                candidate.setdefault("warnings", {})[case["id"]] = warnings
                candidate.setdefault("context", {})[case["id"]] = released.context_differences(data, entry, cfg, max_dom)
            candidates.append(candidate)
        except ValueError as exc:
            rejected.append({"id": candidate["id"], "reason": str(exc)})
    if not candidates:
        raise ValueError("no admitted WRF-valid configurations; CPU-only/catalog status is insufficient")
    cloud_config = None
    if raw["objective"] == "cloud":
        from .cloud import load_config
        cloud_config = load_config(raw["cloud"], resolve)
        stations, obs = None, None
    else:
        stations, obs = resolve(raw["stations"]), resolve(raw["observations"])
    train_count = sum(c["split"] == "train" for c in cases)
    test_count = sum(c["split"] == "test" for c in cases)
    study = {"schema": raw["schema"], "region": raw["region"], "objective": raw["objective"], "variables": variables,
             "scales": scales, "hours": hours, "budget_case_hours": budget, "test_start": split.isoformat(), "e41": e41,
             "cases": sorted(cases, key=lambda c: c["date"]), "candidates": sorted(candidates, key=lambda c: c["id"]),
             "stations": str(stations) if stations else None, "observations": str(obs) if obs else None,
             "sampling": raw.get("sampling", {}),
             "model_sha256": source, "inputs": [identity(p) for p in (path, stations, obs) if p is not None]+evidence+
                 [c["source_namelist"] for c in cases]+[identity(released.RELEASED_PATH)],
             "rejected": rejected, "release_list": {"release": data["release"], "checkpoint": data["checkpoint"]},
             "qualification": ("released v0.4.0 FINAL D6 PASS configurations (exact tuples); the study case context "
                               "differs from the qualifying runs as listed per candidate; no regional skill inferred")}
    if cloud_config is not None:
        study["cloud"] = cloud_config
        study["inputs"].extend(cloud_config["truth_identity"])
    study["rounds"], study["planned_case_hours"] = schedule(len(candidates), train_count, test_count, hours, budget,
                                                            rungs=raw.get("rungs"))
    study["program_groups"] = program_groups(study)
    study["resources"] = plan_resources(study)
    return study


def schedule(count, train_dates, test_dates, hours, budget, *, rungs=None):
    """Double training dates, halve candidates; reserve all test dates for winner.

    Explicit ``rungs`` = [{"train_dates": n, "keep": k}, ...]: rung r scores the survivors on the first
    n training dates (strictly increasing, the last uses all of them) and keeps the best k (last k = 1).
    """
    if rungs is not None:
        if not isinstance(rungs, list) or not rungs or any(not isinstance(r, dict) or set(r) != {"train_dates", "keep"} for r in rungs):
            raise ValueError("rungs must be a nonempty list of {train_dates, keep}")
        rounds, used, previous = [], test_dates*hours, 0
        for number, rung in enumerate(rungs):
            target, keep = _positive_int(rung["train_dates"], "rung train_dates"), _positive_int(rung["keep"], "rung keep")
            if target <= previous or target > train_dates:
                raise ValueError("rung train_dates must increase and stay within the training dates")
            if keep > count:
                raise ValueError("a rung cannot keep more configurations than it scores")
            used += count*(target-previous)*hours
            rounds.append({"candidates": count, "train_dates": target, "keep": keep})
            previous, count = target, keep
        if previous != train_dates or rounds[-1]["keep"] != 1:
            raise ValueError("the last rung must use every training date and keep exactly one configuration")
    else:
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


def record_count_independent():
    """True once the step programs see only an active record window (v0.4.1 record windows, lane w41-bdyrec)."""
    import importlib.util
    return importlib.util.find_spec("gpuwrf.integration.record_windows") is not None


def program_groups(study):
    """Compiled step programs per configuration: one per boundary record-count signature in v0.4.0."""
    groups = {}
    for case in study["cases"]:
        groups.setdefault(json.dumps(case["records"], sort_keys=True), []).append(case["id"])
    independent = record_count_independent()
    return {"signatures": [{"records": json.loads(k), "cases": v} for k, v in sorted(groups.items())],
            "record_count_independent": independent,
            "cold_programs_upper_bound": len(study["candidates"])*(1 if independent else len(groups)),
            "note": ("step programs see only the active boundary record window: one program per configuration"
                     if independent else
                     "[I] v0.4.0 step programs take every wrfbdy/wrflowinp record as a traced shape: each distinct "
                     "record signature is a separate compiled program per configuration; generate all case inputs over "
                     "the same window length to share programs")}


def plan_resources(study):
    from .resources import admission, member_pool

    members = [r["candidates"]*(r["train_dates"]-p) for r, p in
               zip(study["rounds"], [0]+[r["train_dates"] for r in study["rounds"][:-1]])]
    widest = max(members+[sum(c["split"] == "test" for c in study["cases"])])
    if study["cases"][0]["max_dom"] > 1:
        return {"sizing": "launcher C-auto memory plans (nested)", "members_per_rung": members}
    pool = member_pool(study["cases"][0]["geometry"])
    return {"sizing": "fixed per-member pool (--pool-gib; single-domain has no C-auto plan)", "member": pool,
            "members_per_rung": members, "widest_batch": widest,
            "admission_dev_rtx5090": admission(pool["need_gib"], members=widest)}


def evaluate(study, fetch, *, prefetch=None, on_round=None):
    """fetch(candidate, cases) is lazy; losing configurations never read test truth.

    prefetch(pairs), if given, is called once per rung with every missing (candidate, case) pair
    of that rung (and once for the winner's held-out dates), so a launcher can run all
    configurations of a rung in one batch. on_round(record) receives each finished rung.
    """
    if study["objective"] == "cloud":
        from .cloud import evaluate
        return evaluate(study, fetch, prefetch=prefetch, on_round=on_round)
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

    def missing(cands, cases):
        return [(c, k) for c in cands for k in cases if (c["id"], k["id"]) not in cache]

    for number, spec in enumerate(study["rounds"]):
        dates = train[:spec["train_dates"]]
        if prefetch is not None:
            pairs = missing(candidates, dates)
            if pairs:
                prefetch(pairs)
        frames = {c["id"]: collect(c, dates) for c in candidates}
        keys = KEYS+["case_id"]
        support = None
        for frame in frames.values():
            rows = frame[keys]
            support = rows if support is None else support.merge(rows, on=keys, validate="one_to_one")
        scores = []
        for candidate in candidates:
            frame = frames[candidate["id"]].merge(support, on=keys, validate="one_to_one")
            if set(frame.case_id) != {c["id"] for c in dates}:
                raise ValueError("one or more dates have no common support across candidates")
            scores.append({"id": candidate["id"], **metrics(frame, study["variables"], study["scales"])})
        scores.sort(key=lambda s: (s["loss"], s["id"]))
        keep = spec.get("keep", max(1, math.ceil(len(scores)/2)))
        record = {"round": number, "train_dates": spec["train_dates"], "dates": [c["id"] for c in dates],
                  "scores": scores, "common_rows": len(support), "keep": keep,
                  "survivors": [s["id"] for s in scores[:keep]]}
        rounds.append(record)
        if on_round is not None:
            on_round(record)
        survivors = set(record["survivors"])
        candidates = [c for c in candidates if c["id"] in survivors]
    winner = next(c for c in study["candidates"] if c["id"] == rounds[-1]["scores"][0]["id"])
    if prefetch is not None:
        pairs = missing([winner], test)
        if pairs:
            prefetch(pairs)
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
    stations = read_stations(study["stations"]) if study["objective"] != "cloud" else None
    study = copy.deepcopy(study)
    study["inputs"].append(identity(inventory_path))
    def fetch(candidate, cases):
        results = {}
        for case in cases:
            files = [inventory_path.parent/p for p in inventory["forecasts"][candidate["id"]][case["id"]]]
            rows, audit = ((files, {"format": "cloud wrfout"}) if study["objective"] == "cloud"
                           else forecast(files, stations, **study["sampling"]))
            results[case["id"]] = (rows, audit, [identity(p) for p in files])
        return results
    return evaluate(study, fetch)


def program_key(candidate):
    """Compiled program of a candidate: its configuration, or the shared base program of continuous members (traced
    surface controls change values, never the program)."""
    return candidate.get("program", candidate["id"])


def cold_first(pairs, compiled):
    """Two waves: one member per (configuration, record signature) whose program is not compiled yet, then the
    rest. Members of the second wave load the program the first wave compiled instead of compiling it again in
    parallel (same HLO compiled twice; each cold compile also needs several GB of host RAM)."""
    first, rest, seen = [], [], set(compiled)
    for candidate, case in pairs:
        key = (program_key(candidate), json.dumps(case["records"], sort_keys=True))
        (rest if key in seen else first).append((candidate, case))
        seen.add(key)
    return [wave for wave in (first, rest) if wave]


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+f".{os.getpid()}.new")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=1, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def bundle_launcher(output, bundle, consumer):
    """Launcher script: members run the sealed bundle's source/caches in a copy-on-write layer (P unchanged)."""
    bundle, consumer = Path(bundle).resolve(strict=True), Path(consumer).resolve(strict=True)
    env = bundle/"bundle.env"
    launcher = bundle/"src"/"scripts"/"run_parallel_cases.sh"
    if not env.is_file() or not launcher.is_file() or not (bundle/"sealed"/"BUNDLE_MANIFEST.json").is_file():
        raise ValueError("bundle needs bundle.env, src/scripts/run_parallel_cases.sh and sealed/BUNDLE_MANIFEST.json")
    scheduler = launcher.with_name("parallel_cases.py")
    if (not scheduler.is_file() or not re.search(r"^OPTIMIZER_REGISTRY_HOOKS = 2\s*(?:#.*)?$",
                                                scheduler.read_text(), flags=re.MULTILINE)
            or not (bundle/"src/src/gpuwrf/optimize/registry.py").is_file()
            or not (bundle/"src/scripts/alisios_registry.py").is_file()):
        raise ValueError("bundle lacks per-member optimizer registry hooks; rebuild a v0.4.5-or-newer bundle "
                         "with scripts/parallel_cases.py and scripts/alisios_registry.py registry support")
    layer = Path(output).resolve()/"bundle_layer"
    if layer == bundle or bundle in layer.parents:
        raise ValueError("optimize output must not lie inside the bundle")
    script = Path(output)/"bundle_launcher.sh"
    script.write_text("#!/usr/bin/env bash\n# gpuwrf optimize: run the launcher inside the sealed bundle (writes go to the layer)\n"
                      "set -euo pipefail\n"
                      f"source '{env}'\n"
                      f"exec '{consumer}' --bundle '{bundle}' --layer '{layer}' -- '{launcher}' \"$@\"\n")
    script.chmod(0o755)
    return script, {"bundle": str(bundle), "bundle_json": identity(bundle/"sealed"/"BUNDLE.json"),
                    "bundle_env": identity(env), "consumer": identity(consumer), "layer": str(layer)}


def _member_receipt(path):
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return {k: receipt.get(k) for k in ("rc", "wall_s", "vmhwm_kb", "vram_peak_mib", "aot_cache_warm",
                                         "aot_cold_executables", "concurrent_at_start", "sizing",
                                         "registry_run_id", "registry_complete")}


def _source_case(member):
    """The optimizer's own prepared input directory of an ensemble member (member manifest source_case)."""
    manifest = json.loads((Path(member["case"])/"member_manifest.json").read_text())
    return Path(manifest["source_case"]).resolve()


def _stat(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns, st.st_ino)


def _admission(plan):
    """What the launcher actually admitted: members started and the most it ran at once (parallel_run.json, else
    the member receipts), plus the free VRAM it saw at its start."""
    root = Path(plan["out_root"])
    receipts = [r for r in (_member_receipt(root/m["id"]/"receipt.json") for m in plan["members"]) if r.get("rc") is not None]
    try:
        run = json.loads((root/"parallel_run.json").read_text())
    except (OSError, ValueError):
        run = {}
    starts = [r["concurrent_at_start"] for r in receipts if r.get("concurrent_at_start") is not None]
    admitted = run.get("peak_concurrency") or (max(starts)+1 if starts else None)
    return {"started": len(receipts), "admitted": admitted, "vram_free_bytes_at_start": run.get("vram_free_bytes_at_start")}


def _member_entry(batch, plan, member, candidate_id, case, program=None):
    out = Path(plan["out_root"])/member["id"]
    files = sorted((out/"wrfout").glob(f"wrfout_{case['domain']}_*"))
    receipt = _member_receipt(out/"receipt.json")
    if not files or receipt.get("rc") != 0:
        return None
    if receipt.get("registry_run_id") and receipt.get("registry_complete") is not True:
        raise ValueError("model finished but its registry terminal event is incomplete; reconcile the original "
                         "invocation and receipt before resume; do not rerun the forecast")
    return {"candidate": candidate_id, "case": case["id"], "batch": batch.name, "member": member["id"],
            "program": program or candidate_id,
            "files": [identity(p) for p in files], "plan": identity(batch/"plan.json"), **receipt}


def _salvage(output, study, ledger):
    """Members that finished (launcher receipt rc 0) inside a batch that never returned are kept."""
    cases = {c["id"]: c for c in study["cases"]}
    # Continuous members are generated while the study runs: the ledger records each before its launch.
    candidates = {c["id"] for c in study.get("candidates", [])} | set(ledger.get("candidates", {}))
    for record in ledger["batches"]:
        if "rc" in record:
            continue
        batch = output/record["batch"]
        try:
            plan = json.loads((batch/"plan.json").read_text())
        except (OSError, ValueError):
            record["rc"], record["note"] = None, "interrupted before launch"
            continue
        for member in plan["members"]:
            try:
                dest = _source_case(member)
            except (OSError, ValueError, KeyError):
                continue
            candidate_id, case_id = dest.parent.name, dest.name.split(".b")[0]
            key = f"{candidate_id}|{case_id}"
            if candidate_id in candidates and case_id in cases and key not in ledger["members"]:
                entry = _member_entry(batch, plan, member, candidate_id, cases[case_id],
                                      ledger.get("candidates", {}).get(candidate_id, {}).get("program"))
                if entry is not None:
                    ledger["members"][key] = entry | {"salvaged": True}
        record["rc"], record["note"] = None, "interrupted; finished members salvaged on resume"


def run(study, output, **kwargs):
    """Serialize optimizer invocations; account for prep, scoring and failed attempts too."""
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs

    refuse_protected_inputs("gpuwrf optimize", [], outputs=[output])
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(output.with_name(output.name + ".optimizer.lock"), os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    prep_wall = kwargs.pop("prep_wall_s", 0.0)
    context = {"started": kwargs.get("clock", time.monotonic)() - prep_wall}
    saved_alarm = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("optimizer output is already running; concurrent resume refused") from exc
        wall = kwargs.get("wall_seconds", 1800)
        if wall is not None and type(wall) in (int, float) and math.isfinite(wall) and wall > 0:
            saved_alarm = (signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL))
            def expire(*_):
                raise Pause("hard wallclock budget exhausted; best admissible result remains in best.json")
            signal.signal(signal.SIGALRM, expire)
            real_deadline = time.monotonic() + wall - prep_wall
            signal.setitimer(signal.ITIMER_REAL, max(0.001, wall-prep_wall))

            @contextmanager
            def model_phase():
                # The scheduler owns deadline teardown while its children exist.
                # Interrupting subprocess.run here would kill only the scheduler.
                signal.setitimer(signal.ITIMER_REAL, 0)
                try:
                    yield
                finally:
                    remaining = real_deadline-time.monotonic()
                    if remaining > 0:
                        signal.setitimer(signal.ITIMER_REAL, remaining)
            context["model_phase"] = model_phase
        return _run(study, output, _context=context, **kwargs)
    except Pause as pause:
        progress = {"status": "PAUSED", "reason": str(pause), "planned_case_hours": study["planned_case_hours"],
                    "best": str(output / "best.json")}
        if output.is_dir():
            _atomic_json(output / "progress.json", progress)
        return progress
    except BaseException as exc:
        if "ledger" in context:
            _atomic_json(output / "progress.json", {"status": "ABORTED" if isinstance(exc, (KeyboardInterrupt, SystemExit))
                         else "FAILED", "reason": str(exc), "best": str(output / "best.json"),
                         "ledger": LEDGER, "scheduled_case_hours": context["ledger"]["scheduled_case_hours"]})
        raise
    finally:
        try:
            if saved_alarm is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, saved_alarm[0])
                signal.setitimer(signal.ITIMER_REAL, *saved_alarm[1])
            if "ledger" in context:
                ledger = context["ledger"]
                elapsed = kwargs.get("clock", time.monotonic)() - context["started"]
                ledger["accounting"]["wall_s"] += elapsed
                ledger["accounting"]["invocations"] += 1
                _atomic_json(output / LEDGER, ledger)
        finally:
            os.close(lock)


def _run(study, output, *, gpus=None, max_parallel=None, launcher=None, pool_gib=None, resume=False,
        chunk_minutes=None, member_wall_s=None, bundle=None, bundle_consumer=None, member_env=None,
        host_gb_per_member=None, clock=time.monotonic, driver=None, wave_mode=True, registry=None,
        wall_seconds=1800, max_retries=1, profile_points=None, _context):
    """No JAX initialization here; the existing ensemble launcher owns GPU execution.

    Every rung's new (configuration, date) members of ALL surviving configurations go to the launcher
    together (split only to fit a chunk). Completed members are recorded in ledger.json;
    ``resume`` continues an interrupted or paused study in the same directory. ``chunk_minutes``
    stops launching once the next batch's estimated wall would exceed the budget (status PAUSED).

    ``driver(prefetch=, fetch=, files=, on_round=, output=, ledger=)`` replaces the successive-halving
    evaluation (continuous studies): candidates may then carry ``program`` (shared compiled program) and
    ``controls`` (identity of a Track P control file copied into every member input directory).
    """
    from gpuwrf.ens import workflows
    from gpuwrf.verify.observations import read_stations
    from .data import forecast

    max_parallel = _positive_int(max_parallel, "max_parallel") if max_parallel is not None else None
    output = Path(output).resolve()
    if type(wave_mode) is not bool:
        raise ValueError("wave_mode must be a boolean")
    if resume and not wave_mode and (output / "wave_state.json").exists():
        raise ValueError("wave approvals cannot be disabled when resuming a wave-controlled study")
    canonical = json.loads(json.dumps(study, allow_nan=False))
    if wall_seconds is not None and (isinstance(wall_seconds, bool) or not math.isfinite(wall_seconds)
                                     or wall_seconds <= 0):
        raise ValueError("wall_seconds must be positive and finite")
    if max_retries is not None and (type(max_retries) is not int or max_retries < 0):
        raise ValueError("max_retries must be a nonnegative integer")
    if profile_points is not None:
        if (not isinstance(profile_points, list) or not profile_points
                or any(not isinstance(p, dict) or set(p) != {"id", "region", "domain", "j", "i"}
                       or not isinstance(p["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", p["id"])
                       or p["region"] not in ("north_windward", "sea")
                       or not isinstance(p["domain"], str) or not re.fullmatch(r"d[0-9]{2}", p["domain"])
                       or any(type(p[k]) is not int or p[k] < 0 for k in ("j", "i")) for p in profile_points)
                or len({p["id"] for p in profile_points}) != len(profile_points)):
            raise ValueError("profile points require unique id, north_windward/sea region, dNN domain and nonnegative integer j/i")
        try:
            from .diagnostics import write_profile_report  # noqa: F401
        except ImportError as exc:
            raise ValueError("profile diagnostics support is unavailable in this build") from exc
    if resume:
        saved = output/"study.json"
        saved_study = json.loads(saved.read_text()) if saved.is_file() else None
        if isinstance(saved_study, dict) and saved_study.get("optimizer_implementation") != canonical.get("optimizer_implementation"):
            raise ValueError("resume EKI implementation/NumPy version changed or was not recorded; "
                             "use the original optimizer implementation or a new output directory")
        if saved_study != canonical:
            raise ValueError("resume needs the same study (inputs, configurations, rungs) in an existing output directory")
        ledger = json.loads((output/LEDGER).read_text())
        _salvage(output, study, ledger)
        _atomic_json(output/LEDGER, ledger)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"study.json", canonical)
        ledger = {"schema": "gpuwrf.optimize.ledger.v1", "members": {}, "batches": [], "scheduled_case_hours": 0,
                  "candidates": {}, "attempts": {}, "max_retries": max_retries,
                  "profile_points": profile_points,
                  "accounting": {"wall_s": 0.0, "invocations": 0, "prepare_s": 0.0, "model_s": 0.0,
                                 "extract_qc_s": 0.0}}
        _atomic_json(output/LEDGER, ledger)
    if ledger.get("max_retries") != max_retries or "attempts" not in ledger:
        raise ValueError("resume needs the same retry cap and a cap-aware optimizer ledger")
    if ledger.get("profile_points") != profile_points:
        raise ValueError("resume needs the same profile point observer")
    _context["ledger"] = ledger
    # Registry policy is immutable across optimizer restarts, just like the study.
    policy = {"forcing": "OTHER", **(registry or {})}
    policy_path = output / "registry_policy.json"
    if resume:
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("resume needs the same registry policy")
    else:
        _atomic_json(policy_path, policy)
    from .waves import Waves
    waves = Waves(output, canonical, resume=resume, atomic_json=_atomic_json,
                  profile_points=profile_points) if wave_mode else None
    if waves is not None and waves.state.get("stopped"):
        progress = {"status": "STOPPED", "reason": "external reviewer STOP", "planned_case_hours": study["planned_case_hours"]}
        _atomic_json(output / "progress.json", progress)
        return progress
    if resume and (output / "result.json").is_file():
        return json.loads((output / "result.json").read_text())
    single = study["cases"][0]["max_dom"] == 1
    if single and pool_gib is None:
        pool_gib = study.get("resources", {}).get("member", {}).get("pool_gib")
        if pool_gib is None:
            raise ValueError("single-domain members need a per-member pool (--pool-gib)")
    if bundle is not None:
        if launcher is not None:
            raise ValueError("--bundle selects the launcher; do not combine with an explicit launcher")
        if bundle_consumer is None:
            bundle_consumer = Path(__file__).resolve().parents[3]/"scripts"/"release"/"bundle_consumer.sh"
        launcher, bundle_record = bundle_launcher(output, bundle, bundle_consumer)
    else:
        bundle_record = None
    stations = read_stations(study["stations"]) if study.get("objective") != "cloud" and study["stations"] else None
    started = _context["started"]
    deadline = None if wall_seconds is None else started + wall_seconds
    budget_s = None if chunk_minutes is None else float(chunk_minutes)*60
    launched_here = [0]
    default_wall = float(member_wall_s) if member_wall_s else float(study["hours"])*60+900  # [I] cold compile + run
    environ = member_env if member_env is not None else os.environ
    cpu_members = environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"  # launcher: no VRAM admission
    # Input files may not change while members run either: same size, mtime and inode as when this run started.
    stats = {f["path"]: _stat(f["path"]) for case in study["cases"] for f in case["forcing"]}

    def check_wall():
        if deadline is not None and clock() >= deadline:
            raise Pause("hard wallclock budget exhausted; best admissible result remains in best.json")

    def accounting():
        return {**ledger["accounting"], "wall_s": ledger["accounting"]["wall_s"] + clock()-started,
                "wall_seconds_this_invocation": wall_seconds,
                "wall_seconds_remaining": None if deadline is None else max(0.0, deadline-clock()),
                "case_hours_remaining": study["budget_case_hours"]-ledger["scheduled_case_hours"],
                "max_retries": max_retries, "attempts": ledger["attempts"],
                "compile_extract_failed_time": "included in total wall; compile is included in model_s"}

    @contextmanager
    def measure(name):
        before = clock()
        try:
            yield
        finally:
            ledger["accounting"][name] += clock()-before

    def host_need(warm):
        """Launcher host need per member: explicit, else 1.1 x the largest measured member VmHWM (a fixed pool
        records no host peaks in the launcher, which would otherwise assume 16 GB per member forever). A warm
        wave (programs already compiled) uses the peaks of warm-loaded members when there are any."""
        if host_gb_per_member is not None:
            return float(host_gb_per_member)
        members = [m for m in ledger["members"].values() if m.get("vmhwm_kb")]
        warm_peaks = [m["vmhwm_kb"] for m in members if m.get("aot_cache_warm")]
        peaks = warm_peaks if warm and warm_peaks else [m["vmhwm_kb"] for m in members]
        return round(1.1*max(peaks)/1e6, 2) if peaks else None

    def member_wall(warm):
        members = [m for m in ledger["members"].values() if m.get("wall_s")]
        warm_walls = [m["wall_s"] for m in members if m.get("aot_cache_warm")]
        walls = warm_walls if warm and warm_walls else [m["wall_s"] for m in members]
        return max(walls) if walls else default_wall

    def concurrency(count, warm):
        """Members the launcher will run at once: at most the requested parallelism, 1 without evidence.

        Evidence = the launcher's own admission rules applied to measured needs (host: MemAvailable - 8 GB
        reserve with one cold-compile headroom; VRAM: the free memory the launcher saw at its last start,
        pool + 1 GiB or the measured per-process peak per member) and any batch in which the launcher
        started fewer members at once than it had (a measured admission limit).
        """
        bound = min(count, max_parallel or count)
        records = [b for b in ledger["batches"] if b.get("admitted")]
        limited = [b for b in records if b["admitted"] < b["started"]]
        same = [b for b in limited if b.get("warm") == warm] or limited
        if same:
            bound = min(bound, min(b["admitted"] for b in same))
        need, cold, avail = host_need(warm), host_need(False), mem_available_gb()
        if need is None or avail is None:
            return 1
        bound = min(bound, math.floor((avail-8.0-max(0.0, cold-need))/need))
        if not cpu_members:
            free = [b["vram_free_bytes_at_start"] for b in records if b.get("vram_free_bytes_at_start")]
            peaks = [m["vram_peak_mib"]*2**20 for m in ledger["members"].values() if m.get("vram_peak_mib")]
            vram = max(peaks+([(pool_gib+1.0)*2**30] if single else []), default=None)
            if not free or vram is None:
                return 1
            bound = min(bound, math.floor(free[-1]/vram))
        return max(1, bound)

    def estimate(pairs, warm=False):
        return math.ceil(len(pairs)/concurrency(len(pairs), warm))*member_wall(warm)

    def fit(rest, warm):
        """How many of `rest` the next batch takes so that it ends inside the chunk (Pause when none fits)."""
        if budget_s is None:
            return len(rest)
        per, left = member_wall(warm), budget_s-(clock()-started)
        if per > left:
            if launched_here[0]:
                raise Pause(f"chunk budget {chunk_minutes} min reached")
            return 1  # one member alone already exceeds the chunk: it runs alone, recorded as over_chunk
        return min(len(rest), math.floor(left/per)*concurrency(len(rest), warm))

    def launch_batch(pairs, warm=False):
        check_wall()
        prep_start = clock()
        hours = study["hours"]
        for candidate, case in pairs:
            key = f"{candidate['id']}|{case['id']}"
            if max_retries is not None and ledger["attempts"].get(key, 0) >= max_retries+1:
                raise Pause(f"retry budget exhausted for {key}; best admissible result remains in best.json")
        # Every dispatched attempt counts, including failed and interrupted ones (retries are charged again).
        if ledger["scheduled_case_hours"]+len(pairs)*hours > study["budget_case_hours"]:
            raise ValueError(f"compute budget exhausted before launch: {ledger['scheduled_case_hours']} case-hours "
                             f"dispatched + {len(pairs)*hours} > budget {study['budget_case_hours']}")
        for candidate, case in pairs:
            for f in case["forcing"]:
                if _stat(f["path"]) != stats[f["path"]]:
                    raise ValueError(f"forecast input changed while the study ran: {f['path']}")
        number = len(ledger["batches"])
        batch = output/f"batch-{number:03d}"
        inputs, controls, parameters = [], [], []
        if len({candidate.get("controls") is None for candidate, _ in pairs}) > 1:
            raise ValueError("one batch cannot mix members with and without surface controls")
        if len({candidate.get("runtime_parameters") is None for candidate, _ in pairs}) > 1:
            raise ValueError("one batch cannot mix members with and without runtime parameters")
        for candidate, case in pairs:
            dest = output/"inputs"/candidate["id"]/case["id"]
            if dest.exists():  # an interrupted earlier batch prepared it: same content, fresh copy
                dest = output/"inputs"/candidate["id"]/f"{case['id']}.b{number:03d}"
            dest.mkdir(parents=True)
            cfg = copy.deepcopy(case["config"])
            cfg.setdefault("physics", {}).update(candidate["physics"])
            (dest/"namelist.input").write_text(namelist_text(cfg))
            linked = forecast_inputs(case["input_dir"])
            if [str(p.resolve()) for p in linked] != [f["path"] for f in case["forcing"]]:
                raise ValueError(f"forecast input files of case {case['id']} changed since the study was loaded")
            for source in linked:
                (dest/source.name).symlink_to(source.resolve())
            if candidate.get("controls") is not None:  # a content copy: the member never reads a mutable source
                target = dest/workflows.SURFACE_CONTROLS_NAME
                target.write_bytes(Path(candidate["controls"]["path"]).read_bytes())
                if identity(target)["sha256"] != candidate["controls"]["sha256"]:
                    raise ValueError(f"member control file changed since it was written: {candidate['id']}")
                controls.append(target)
            if candidate.get("runtime_parameters") is not None:
                target = dest/workflows.RUNTIME_PARAMETERS_NAME
                record = candidate["runtime_parameters"]
                target.write_bytes(Path(record["path"]).read_bytes())
                if identity(target)["sha256"] != record["sha256"]:
                    raise ValueError(f"member runtime parameter file changed: {candidate['id']}")
                parameters.append(target)
            ledger.setdefault("candidates", {})[candidate["id"]] = {"program": program_key(candidate),
                                                     "controls": candidate.get("controls"),
                                                     "runtime_parameters": candidate.get("runtime_parameters")}
            inputs.append(dest)
        kwargs = {"pool_gib": pool_gib} if single else {}
        if controls:
            kwargs["surface_controls"] = controls
        if parameters:
            kwargs["runtime_parameters"] = parameters
        host = host_need(warm)
        if host is not None:
            kwargs["host_gb_per_case"] = host
        plan = workflows.prepare("batch", batch, cases=inputs, hours=hours, gpus=gpus,
                                 max_parallel=max_parallel or len(inputs), single_batch=True, **kwargs)
        members = {_source_case(m): m for m in plan["members"]}
        record = {"batch": batch.name, "members": [], "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "pool_gib": kwargs.get("pool_gib"), "host_gb_per_member": kwargs.get("host_gb_per_case"),
                  "bundle": bundle_record, "warm": warm}
        if budget_s is not None:
            record["estimate_s"] = estimate(pairs, warm)
            if record["estimate_s"] > budget_s-(clock()-started):
                record["over_chunk"] = f"one member is estimated at {member_wall(warm):.0f} s, more than the chunk left"
        ledger["batches"].append(record)
        ledger["scheduled_case_hours"] += len(pairs)*hours
        for candidate, case in pairs:
            key = f"{candidate['id']}|{case['id']}"
            ledger["attempts"][key] = ledger["attempts"].get(key, 0) + 1
        _atomic_json(output/LEDGER, ledger)
        launched_here[0] += 1
        t0 = clock()
        from .registry import batch_metadata
        metadata = batch_metadata(study, plan, pairs, inputs, policy)
        metadata_path = batch / "registry.json"
        _atomic_json(metadata_path, metadata)
        child_env = dict(environ, ALISIOS_OPTIMIZE_REGISTRY=str(metadata_path))
        if deadline is not None:
            child_env["ALISIOS_OPTIMIZE_WALL_DEADLINE"] = str(deadline)
        ledger["accounting"]["prepare_s"] += clock()-prep_start
        check_wall()
        t0 = clock()
        try:
            if "model_phase" in _context:
                with _context["model_phase"]():
                    rc = workflows.launch(batch/"plan.json", launcher=launcher, env=child_env)
            else:
                rc = workflows.launch(batch/"plan.json", launcher=launcher, env=child_env)
        finally:
            ledger["accounting"]["model_s"] += clock()-t0
        record["wall_s"] = round(clock()-t0, 3)
        record["rc"] = rc
        record.update(_admission(plan))
        for (candidate, case), dest in zip(pairs, inputs):
            member = members[dest.resolve()]
            record["members"].append(f"{candidate['id']}|{case['id']}")
            # Only members whose launcher receipt says rc 0 count; a failed batch keeps them for --resume.
            entry = _member_entry(batch, plan, member, candidate["id"], case, program_key(candidate))
            if entry is not None:
                ledger["members"][f"{candidate['id']}|{case['id']}"] = entry
        _atomic_json(output/LEDGER, ledger)
        check_wall()
        if rc:
            raise ValueError(f"ensemble batch failed with exit {rc}; no recommendation published")

    def prefetch(pairs):
        check_wall()
        todo = [(c, k) for c, k in pairs if f"{c['id']}|{k['id']}" not in ledger["members"]]
        if not todo:  # resumed: every member of this rung is already recorded
            return
        cases = {c["id"]: c for c in study["cases"]}
        compiled = {(m.get("program", m["candidate"]), json.dumps(cases[m["case"]]["records"], sort_keys=True))
                    for m in ledger["members"].values()}
        for wave in cold_first(todo, compiled):
            warm = all((program_key(c), json.dumps(k["records"], sort_keys=True)) in compiled for c, k in wave)
            rest = list(wave)
            while rest:  # each batch sized to the chunk time left at its start
                take = fit(rest, warm)
                launch_batch(rest[:take], warm=warm)
                rest = rest[take:]
            compiled |= {(program_key(c), json.dumps(k["records"], sort_keys=True)) for c, k in wave}

    def files(candidate, case):
        """The recorded scoring-domain histories of a completed member, re-verified by content."""
        key = f"{candidate['id']}|{case['id']}"
        if key not in ledger["members"]:
            raise ValueError(f"no completed member for {key}")
        entry = ledger["members"][key]
        paths = [Path(f["path"]) for f in entry["files"]]
        if not paths:
            raise ValueError("scheduled forecast produced no scoring-domain wrfouts")
        if [identity(p) for p in paths] != entry["files"]:
            raise ValueError(f"member history changed since it was recorded: {key}")
        return paths, entry

    def fetch(candidate, cases):
        results = {}
        for case in cases:
            paths, entry = files(candidate, case)
            with measure("extract_qc_s"):
                rows, audit = ((paths, {"format": "cloud wrfout"}) if study.get("objective") == "cloud"
                               else forecast(paths, stations, **study["sampling"]))
            results[case["id"]] = (rows, audit, entry["files"]+[entry["plan"]])
        return results

    def on_round(record):
        check_wall()
        members = {}
        for key, entry in ledger["members"].items():
            cand, case = key.split("|")
            if cand in [s["id"] for s in record["scores"]] and case in record["dates"]:
                members[key] = {k: entry.get(k) for k in ("batch", "member", "rc", "wall_s", "vmhwm_kb", "vram_peak_mib",
                                                          "aot_cache_warm", "aot_cold_executables")}
        rungs = output/"rungs"
        rungs.mkdir(exist_ok=True)
        _atomic_json(rungs/f"rung-{record['round']:02d}.json",
                     {"schema": "gpuwrf.optimize.rung.v1", **record, "members": members,
                      "program_groups": study.get("program_groups"), "bundle": bundle_record})
        if waves is not None:
            waves.complete(record["round"], kind="rung", ensemble=[s["id"] for s in record["scores"]],
                           scores=record["scores"], checkpoint={"kind": "rung", "record": record}, ledger=ledger,
                           accounting=accounting(),
                           impact_matrix=record.get("impact_matrix"), admissibility=record.get("admissibility"))

    try:
        if driver is None:
            report = evaluate(study, fetch, prefetch=prefetch, on_round=on_round)
        else:
            report = driver(prefetch=prefetch, fetch=fetch, files=files, on_round=on_round, output=output,
                            ledger=ledger, waves=waves, accounting=accounting, check_wall=check_wall, measure=measure)
    except Pause as pause:
        progress = {"status": "PAUSED", "reason": str(pause), "completed_members": len(ledger["members"]),
                    "batches": len(ledger["batches"]), "scheduled_case_hours": ledger["scheduled_case_hours"],
                    "planned_case_hours": study["planned_case_hours"],
                    "accounting": accounting(),
                    "resume": "gpuwrf optimize run --study <same study> --output <this directory> --resume"}
        _atomic_json(output/"progress.json", progress)
        return progress
    report["scheduled_case_hours"] = ledger["scheduled_case_hours"]
    report["bundle"] = bundle_record
    report["members"] = {k: {kk: v.get(kk) for kk in ("batch", "member", "wall_s", "vmhwm_kb", "vram_peak_mib")}
                         for k, v in ledger["members"].items()}
    write_json(output/"result.json", report)
    _atomic_json(output/"progress.json", {"status": "DONE", "completed_members": len(ledger["members"]),
                                          "scheduled_case_hours": ledger["scheduled_case_hours"]})
    return report
