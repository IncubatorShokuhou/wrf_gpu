"""Minimal registration for the isolated offline station command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register_parser(subparsers):
    parser = subparsers.add_parser("verify", help="verify wrfout/station forecasts against observations (CPU, offline)",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stations", type=Path, required=True, help="station metadata CSV")
    obs = parser.add_mutually_exclusive_group(required=True)
    obs.add_argument("--obs", type=Path, help="canonical or ALISIOS observation CSV")
    obs.add_argument("--fetch-observations", choices=["metar", "isd-lite"], help="explicit optional Meteostat observation download")
    parser.add_argument("--start", help="provider start timestamp with UTC/offset")
    parser.add_argument("--end", help="provider end timestamp with UTC/offset")
    parser.add_argument("--obs-format", choices=["csv", "alisios"], default="csv")
    parser.add_argument("--obs-precip-hours", type=float, help="declared ObsDB precipitation interval ending at ts_utc")
    parser.add_argument("--model", type=Path, nargs="+", required=True, help="single-domain wrfout files or one station forecast CSV")
    parser.add_argument("--compare", type=Path, nargs="+", help="second forecast: CPU-WRF wrfout or driving-model station CSV")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True, help="JSON + markdown + standalone HTML reports")
    parser.add_argument("--method", choices=["nearest", "bilinear"], default="bilinear")
    parser.add_argument("--land-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lapse-rate", type=float, default=0.0, help="opt-in T2 decrease with height, K/m (e.g. 0.0065)")
    parser.add_argument("--wind-height", choices=["as-is", "neutral-log"], default="as-is")
    parser.add_argument("--wind-frame", choices=["grid", "earth"], default="grid", help="input U10/V10 reference frame")
    parser.add_argument("--gust-variable", help="explicit gust field (never infer gust from WSPD10MAX)")
    parser.add_argument("--gust-window-hours", type=float, help="gust window ending at history time; 0 for instantaneous")
    parser.add_argument("--precip-thresholds", type=float, nargs="+", default=[0.1, 1, 5], help="inclusive event thresholds in mm/h")
    parser.set_defaults(func=command)


def command(args):
    from .engine import verify, write_report

    try:
        observation = args.obs
        provider_receipt = None
        if args.fetch_observations:
            from .observations import read_stations
            from .providers import fetch_meteostat
            if not args.start or not args.end:
                raise ValueError("--fetch-observations requires --start and --end")
            if args.obs_format != "csv":
                raise ValueError("provider download uses canonical CSV")
            data, provider_receipt = fetch_meteostat(read_stations(args.stations), args.start, args.end, provider=args.fetch_observations)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            observation = args.output_dir/"observations.csv"
            data.to_csv(observation, index=False)
        report = verify(args.stations, observation, args.model, comparison_paths=args.compare,
                        observation_format=args.obs_format, precip_hours=args.obs_precip_hours,
                        candidate_name=args.candidate_name, baseline_name=args.baseline_name,
                        thresholds=args.precip_thresholds, method=args.method, land_aware=args.land_aware,
                        lapse_rate=args.lapse_rate, wind_height=args.wind_height, wind_frame=args.wind_frame,
                        gust_variable=args.gust_variable, gust_hours=args.gust_window_hours)
        if provider_receipt:
            report["observation_provider"] = provider_receipt
        write_report(report, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        print(f"gpuwrf verify: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "report": str(args.output_dir/"verification.html")}, allow_nan=False))
    return 0
