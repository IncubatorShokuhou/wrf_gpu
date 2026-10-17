"""Backend-dark command registration; offline work imports NumPy/pandas lazily."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def register_parser(subparsers):
    optimizer = subparsers.add_parser("optimize", help="select admitted regional physics with a date-held-out study")
    actions = optimizer.add_subparsers(dest="optim_action", required=True)
    source = actions.add_parser("identity", help="print model source/version binding for qualification admissions")
    source.set_defaults(func=optimize_command)
    for name in ("plan", "score", "run"):
        parser = actions.add_parser(name)
        parser.add_argument("--study", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True,
                            help="new JSON file (plan/score), new directory (run)")
        if name == "score":
            parser.add_argument("--forecasts", type=Path, required=True)
        if name == "run":
            parser.add_argument("--gpus", help="physical indices or all; delegated to ensemble scheduler")
            parser.add_argument("--max-parallel", type=int, default=1)
        parser.set_defaults(func=optimize_command)
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


def optimize_command(args):
    from .study import load_study, score, run
    from .data import write_json
    try:
        if args.optim_action == "identity":
            from .study import model_digest
            from gpuwrf import __version__
            print(json.dumps({"model_sha256": model_digest(), "gpuwrf_version": __version__}))
            return 0
        if args.output.exists():
            raise ValueError("output exists; use a new destination")
        study = load_study(args.study)
        if args.optim_action == "plan":
            result = study
        elif args.optim_action == "score":
            result = score(study, args.forecasts)
        else:
            result = run(study, args.output, gpus=args.gpus, max_parallel=args.max_parallel)
        if args.optim_action != "run":
            write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "status": result.get("status", "PLANNED"),
                          "planned_case_hours": result["planned_case_hours"]}, allow_nan=False))
        return 0
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
