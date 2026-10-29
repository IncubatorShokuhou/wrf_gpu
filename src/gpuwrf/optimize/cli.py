"""Backend-dark command registration; offline work imports NumPy/pandas lazily."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def register_parser(subparsers):
    optimizer = subparsers.add_parser("optimize", help="select admitted regional physics with a date-held-out study; "
                                      "continuous studies (schema gpuwrf.optimize.continuous.v1) calibrate traced "
                                      "surface parameters by ensemble Kalman inversion")
    actions = optimizer.add_subparsers(dest="optim_action", required=True)
    assessment = actions.add_parser("assess", help="joint cloud/wind/T2/RH score and reference/control guard")
    assessment.add_argument("--assessment", required=True, type=Path)
    assessment.add_argument("--output", required=True, type=Path, help="new JSON and adjacent .md impact matrix")
    assessment.set_defaults(func=assessment_command)
    source = actions.add_parser("identity", help="print model source/version binding for qualification admissions")
    source.set_defaults(func=optimize_command)
    listing = actions.add_parser("released", help="physics configurations released for optimize (FINAL D6 PASS tuples)")
    listing.add_argument("--max-dom", type=int, help="only configurations for this domain count")
    listing.add_argument("--e41", choices=("any", "PASS"), default="any", help="PASS: only tuples with an E41 PASS run")
    listing.add_argument("--json", action="store_true", help="print the machine-readable entries")
    listing.set_defaults(func=optimize_command)
    best = actions.add_parser("best", help="retrieve the best externally assessed admissible solution so far")
    best.add_argument("--output", type=Path, required=True, help="optimizer run directory")
    best.set_defaults(func=optimize_command)
    for name in ("plan", "score", "run"):
        parser = actions.add_parser(name)
        parser.add_argument("--study", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True,
                            help="new JSON file (plan/score), new directory (run; existing with --resume)")
        if name == "score":
            parser.add_argument("--forecasts", type=Path, required=True)
        if name == "run":
            parser.add_argument("--gpus", help="physical indices or all; delegated to ensemble scheduler")
            parser.add_argument("--max-parallel", type=int, default=None,
                                help="members at once (default: every member of a batch; VRAM/host admission decides)")
            parser.add_argument("--pool-gib", type=float, default=None,
                                help="single-domain member pool (default: the plan's resources estimate)")
            parser.add_argument("--resume", action="store_true", help="continue a paused/interrupted run in --output")
            parser.add_argument("--registry", type=Path, help="registry policy JSON (forcing, atlas_dir, smoke, by, producer); "
                                "default: OTHER forcing, live registry. Tests require a private atlas and smoke=true")
            parser.add_argument("--wall-seconds", type=float, default=1800,
                                help="hard launch wall budget per invocation (default 1800 s; teardown is additional)")
            parser.add_argument("--max-retries", type=int, default=1,
                                help="maximum additional attempts per candidate/date, including across resumes")
            parser.add_argument("--profile-points", type=Path,
                                help="JSON list of north-windward/sea grid points for offline held-history profile reports")
            parser.add_argument("--chunk-minutes", type=float, default=None,
                                help="stop launching new batches after this wall budget (exit 3, then --resume)")
            parser.add_argument("--host-gb-per-member", type=float, default=None,
                                help="launcher host-RAM need per member (default: 1.1 x the largest measured member peak)")
            parser.add_argument("--member-wall-s", type=float, default=None,
                                help="estimated wall per member before any was measured (chunk planning)")
            parser.add_argument("--bundle", type=Path, default=None,
                                help="sealed compile bundle P: members run P's source and caches in a private layer")
            parser.add_argument("--bundle-consumer", type=Path, default=None,
                                help="bundle_consumer.sh (default: scripts/release/bundle_consumer.sh of this checkout)")
        parser.set_defaults(func=optimize_command)
    from .contrasts import register_parser as register_contrasts
    register_contrasts(actions, optimize_command)
    calibration = subparsers.add_parser("calibrate", help="station bias/linear MOS with chronological cross-validation")
    actions = calibration.add_subparsers(dest="cal_action", required=True)
    fit = actions.add_parser("fit")
    fit.add_argument("--stations", type=Path, required=True)
    fit.add_argument("--obs", type=Path, required=True)
    fit.add_argument("--forecasts", type=Path, required=True, help="past canonical station forecast CSV")
    fit.add_argument("--method", choices=["bias", "mos"], default="bias")
    fit.add_argument("--min-train-dates", type=int, default=3)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(func=calibrate_command)
    apply = actions.add_parser("apply")
    apply.add_argument("--model", type=Path, required=True)
    apply.add_argument("--forecasts", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.set_defaults(func=calibrate_command)


def assessment_command(args):
    from .multiobjective import command
    return command(args)


def driver_cpu_only():
    """The optimizer process never needs a JAX backend (E50/E132): pin it to the CPU platform.

    Returns the caller's environment for the model members, who must get the caller's platform back.
    """
    members = dict(os.environ)
    os.environ["JAX_PLATFORMS"] = "cpu"
    return members


def optimize_command(args):
    member_env = driver_cpu_only()
    try:
        return _optimize(args, member_env)
    finally:
        if "JAX_PLATFORMS" in member_env:
            os.environ["JAX_PLATFORMS"] = member_env["JAX_PLATFORMS"]
        else:
            os.environ.pop("JAX_PLATFORMS", None)


def _optimize(args, member_env):
    prepared_at = time.monotonic()
    from .study import PAUSED, load_study, score, run
    from .data import write_json
    try:
        if args.optim_action == "contrasts":
            from .contrasts import command
            return command(args, member_env)
        if args.optim_action == "identity":
            from .study import model_digest
            from gpuwrf import __version__
            print(json.dumps({"model_sha256": model_digest(), "gpuwrf_version": __version__}))
            return 0
        if args.optim_action == "best":
            from .continuous import read_admitted
            _, text = read_admitted(args.output / "best.json")
            print(text)
            return 0
        if args.optim_action == "released":
            from . import released
            data = released.load()
            if args.json:
                entries = [e for e in data["configurations"] if (args.max_dom is None or e["max_dom"] == args.max_dom)
                           and (args.e41 == "any" or e["best_E41"] == "PASS")]
                print(json.dumps({"release": data["release"], "rule": data["rule"], "configurations": entries}, indent=1))
            else:
                print(f"# {data['release']}: {data['rule']}")
                print("\n".join(released.table(data, max_dom=args.max_dom, e41=args.e41)))
            return 0
        if args.output.exists() and not (args.optim_action == "run" and args.resume):
            raise ValueError("output exists; use a new destination (run: --resume continues it)")
        from . import continuous
        admitted, text = continuous.read_admitted(args.study)  # ALISIOS preflight before any byte is read
        if continuous.is_continuous(text):  # EKI over traced surface controls (one program, many values)
            if args.optim_action == "score":
                raise ValueError("score takes discrete studies; continuous studies run their members (optimize run)")
            study, run = continuous.load_continuous(admitted, text=text), continuous.run
        else:
            study = load_study(args.study)
        if args.optim_action == "plan":
            result = study
        elif args.optim_action == "score":
            result = score(study, args.forecasts)
        else:
            registry_path = getattr(args, "registry", None)
            registry = None
            points = None
            if getattr(args, "profile_points", None) is not None:
                _, points_text = continuous.read_admitted(args.profile_points)
                points = json.loads(points_text)
                if not isinstance(points, list) or not points:
                    raise ValueError("profile-points must contain a nonempty JSON list")
            if registry_path is not None:
                _, registry_text = continuous.read_admitted(registry_path)
                registry = json.loads(registry_text)
                if not isinstance(registry, dict):
                    raise ValueError("registry policy must be a JSON object")
            result = run(study, args.output, gpus=args.gpus, max_parallel=args.max_parallel, pool_gib=args.pool_gib,
                         resume=args.resume, chunk_minutes=args.chunk_minutes, member_wall_s=args.member_wall_s,
                         bundle=args.bundle, bundle_consumer=args.bundle_consumer, member_env=member_env,
                         host_gb_per_member=args.host_gb_per_member, registry=registry,
                         wall_seconds=getattr(args, "wall_seconds", 1800), max_retries=getattr(args, "max_retries", 1),
                         prep_wall_s=time.monotonic()-prepared_at, profile_points=points)
        if args.optim_action != "run":
            write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "status": result.get("status", "PLANNED"),
                          "planned_case_hours": result["planned_case_hours"]}, allow_nan=False))
        return PAUSED if result.get("status") == "PAUSED" else 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"gpuwrf optimize: error: {exc}", file=sys.stderr)
        return 2


def calibrate_command(args):
    from .calibration import fit, apply
    from .data import write_json
    try:
        if args.cal_action == "fit":
            model = fit(args.stations, args.obs, args.forecasts, method=args.method, min_train_dates=args.min_train_dates)
            write_json(args.output, model)
            result = {"output": str(args.output), "groups": len(model["coefficients"]), "status": model["status"]}
        else:
            result = apply(args.model, args.forecasts, args.output)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"gpuwrf calibrate: error: {exc}", file=sys.stderr)
        return 2
