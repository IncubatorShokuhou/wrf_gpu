"""Lightweight registration; actual data readers load only for an ens command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def register_parser(subparsers):
    parser = subparsers.add_parser("ens", help="offline ensemble products, verification and prepared-case workflows")
    actions = parser.add_subparsers(dest="ens_action", required=True)
    grid = actions.add_parser("products", help="stream mean/spread/quantiles/exceedance from member wrfouts")
    grid.add_argument("--members", type=Path, required=True, help="JSON inventory of member IDs and wrfout paths")
    grid.add_argument("--variable", required=True, help="2D/3D WRF mass-grid scalar (e.g. T2)")
    grid.add_argument("--output", type=Path, required=True, help="new ensemble NetCDF")
    grid.add_argument("--alignment", choices=["exact", "intersection"], default="exact")
    grid.add_argument("--thresholds", type=float, nargs="+", default=[0.0], help="strict X>threshold, input variable units")
    grid.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    verification = actions.add_parser("verify", help="verify ensemble on gpuwrf verify station/interval support")
    verification.add_argument("--members", type=Path, required=True, help="JSON inventory, wrfouts or one station CSV per member")
    verification.add_argument("--stations", type=Path, required=True)
    verification.add_argument("--obs", type=Path, required=True)
    verification.add_argument("--obs-format", choices=["csv", "alisios"], default="csv")
    verification.add_argument("--obs-precip-hours", type=float)
    verification.add_argument("--variable", choices=["t2", "wind", "gust", "precip_rate"], default="t2")
    verification.add_argument("--thresholds", type=float, nargs="+", default=[0.0], help="strict >; degC, m/s, mm/h by variable")
    verification.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    verification.add_argument("--output-dir", type=Path, required=True, help="new directory for JSON/CSV + SVG figures")
    verification.add_argument("--method", choices=["nearest", "bilinear"], default="bilinear")
    verification.add_argument("--land-aware", action=argparse.BooleanOptionalAction, default=True)
    verification.add_argument("--lapse-rate", type=float, default=0.0)
    verification.add_argument("--wind-height", choices=["as-is", "neutral-log"], default="as-is")
    verification.add_argument("--wind-frame", choices=["grid", "earth"], default="grid")
    verification.add_argument("--gust-variable")
    verification.add_argument("--gust-window-hours", type=float)
    prep = actions.add_parser("prepare", help="prepare shared-input physics, lagged, multi-date or preset member plans")
    prep.add_argument("--mode", choices=["physics", "lagged", "batch", "stochastic", "multiphysics"], required=True,
                      help="stochastic: N members with WRF SKEBS/SPPT[/SPP] and distinct nens; multiphysics: N rows "
                           "of the qualified physics table (docs/ensemble.md)")
    prep.add_argument("--case", type=Path)
    prep.add_argument("--namelists", type=Path, nargs="+")
    prep.add_argument("--cases", type=Path, nargs="+")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--hours", type=int, default=24)
    prep.add_argument("--gpus", help="all or physical nvidia-smi indices, e.g. 0,1; scheduler M1")
    prep.add_argument("--max-parallel", type=int, default=64)
    prep.add_argument("--max-per-gpu", type=int)
    prep.add_argument("--pool-gib", type=float)
    prep.add_argument("--checkpoint-interval-steps", type=int, default=0)
    prep.add_argument("--retry-failed", type=int, default=0)
    prep.add_argument("--members", type=int, help="preset member count")
    prep.add_argument("--schemes", type=lambda v: tuple(x.strip().lower() for x in v.split(",") if x.strip()),
                      help="comma list of skebs,sppt,spp (stochastic default skebs,sppt; multiphysics default none)")
    prep.add_argument("--lengthscale-km", type=float, help="Gaussian SPPT/SPP pattern length scale for every active pattern")
    prep.add_argument("--nens-start", type=int, default=1, help="WRF nens of the first member (then +1 per member)")
    prep.add_argument("--physics-rows", type=lambda v: tuple(x.strip().upper() for x in v.split(",") if x.strip()),
                      help="multiphysics table rows, e.g. P00,P02,P05 (default: the first --members rows)")
    run = actions.add_parser("launch", help="launch a prepared plan through scripts/run_parallel_cases.sh")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true", help="scheduler admission preview (GPU metadata queried under lock)")
    for action in (grid, verification, prep, run):
        action.set_defaults(func=command)


def command(args):
    try:
        if args.ens_action == "products":
            from .wrfout import write_products
            result = write_products(args.members, args.variable, args.output, alignment=args.alignment,
                                    thresholds=args.thresholds, quantiles=args.quantiles)
        elif args.ens_action == "verify":
            from .verification import verify_members, write_verification
            report, table = verify_members(args.members, args.stations, args.obs, variable=args.variable,
                thresholds=args.thresholds, quantiles=args.quantiles, observation_format=args.obs_format,
                precip_hours=args.obs_precip_hours, method=args.method, land_aware=args.land_aware,
                lapse_rate=args.lapse_rate, wind_height=args.wind_height, wind_frame=args.wind_frame,
                gust_variable=args.gust_variable, gust_hours=args.gust_window_hours)
            write_verification(report, table, args.output_dir)
            result = {"status": report["status"], "n": report["n"], "output": str(args.output_dir)}
        elif args.ens_action == "prepare":
            from .workflows import prepare
            plan = prepare(args.mode, args.output_dir, case=args.case, namelists=args.namelists, cases=args.cases,
                hours=args.hours, gpus=args.gpus, max_parallel=args.max_parallel, max_per_gpu=args.max_per_gpu,
                pool_gib=args.pool_gib, checkpoint_interval_steps=args.checkpoint_interval_steps, retry_failed=args.retry_failed,
                members=args.members, schemes=args.schemes, lengthscale_km=args.lengthscale_km,
                nens_start=args.nens_start, physics_rows=args.physics_rows)
            result = {"mode": plan["mode"], "members": len(plan["members"]), "plan": str(args.output_dir/"plan.json")}
        else:
            from .workflows import launch
            return launch(args.plan, dry_run=args.dry_run)
    except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"gpuwrf ens: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0
