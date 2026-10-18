"""``gpuwrf da`` — offline ensemble data assimilation (LETKF), experimental, opt-in, d01 only.

    gpuwrf da check   --config da.json          validate + list every input (nothing is opened)
    gpuwrf da analyse --config da.json --out D  one analysis time: receipts, optional analysis wrfinput(s)
    gpuwrf da cycle   ...                       forecast/analysis cycles (GPU member driver, cycle_loop)

The analysis is host-only (NumPy/SciPy/netCDF4); ``check`` and ``analyse`` import no JAX and never touch a
GPU. Exit codes: 0 done (every requested output written), 1 done but a requested output was refused
(see the receipts), 2 invalid configuration / missing or forbidden input. See ``docs/da.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _short(summary):
    out = {"t_a": summary.get("t_a"), "config_sha256": summary.get("config_sha256"),
           "n_obs": summary.get("n_obs"), "wall_s": summary.get("wall_s"), "experiments": {}}
    for name, exp in summary.get("experiments", {}).items():
        row = {"admissible": exp["admissible"], "assimilated": exp["assimilated"], "withheld_ok": exp["withheld_ok"]}
        if exp.get("refusal_reasons"):
            row["refusal_reasons"] = exp["refusal_reasons"]
        for key in ("analysis_wrfinput", "member_cycle_manifest"):
            if exp.get(key):
                row[key] = exp[key]
        out["experiments"][name] = row
    return out


def _refused_outputs(summary):
    refused = []
    for name, exp in summary.get("experiments", {}).items():
        for key in ("analysis_wrfinput", "member_cycle_manifest"):
            if isinstance(exp.get(key), dict) and exp[key].get("written") is False:
                refused.append(f"{name}:{key}")
    return refused


def cmd_check(args) -> int:
    from . import api, config, governance

    try:
        cfg, policy, plan = api.prepare(config.load(args.config))
    except (config.ConfigError, governance.PlanError, FileNotFoundError, LookupError) as exc:
        print(f"gpuwrf da check: error: {exc}", file=sys.stderr)
        return 2
    roles = {}
    for entry in plan.entries:
        roles[entry.role] = roles.get(entry.role, 0) + 1
    report = {"status": "OK", "config": str(Path(args.config).resolve()), "config_sha256": cfg.sha256,
              "t_a": cfg.data["t_a"], "domain": cfg.domain, "ensemble": cfg.data["ensemble"]["kind"],
              "governance": policy.name, "planned_inputs": roles, "dropped_candidates": len(plan.dropped),
              "metadata_receipts": plan.notes.get("metadata_receipts"),
              "experiments": [e["name"] for e in cfg.data["experiments"]]}
    if args.json:
        report["read_plan"] = plan.to_dict()
    print(json.dumps(report, indent=1, default=str))
    return 0


def cmd_analyse(args) -> int:
    from . import api, config, governance

    try:
        cfg = config.load(args.config)
        summary = api.analyse_case(cfg, args.out)
    except (config.ConfigError, governance.PlanError, FileExistsError, FileNotFoundError, LookupError) as exc:
        print(f"gpuwrf da analyse: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_short(summary), indent=1, default=float))
    refused = _refused_outputs(summary)
    if refused:
        print(f"gpuwrf da analyse: requested output refused: {refused} (see receipts)", file=sys.stderr)
        return 1
    return 0


def cmd_cycle_unavailable(args) -> int:
    print("gpuwrf da cycle: error: the cycle driver is not available in this build", file=sys.stderr)
    return 2


def register(subparsers):
    """Add ``gpuwrf da`` (registration imports nothing heavy)."""
    da = subparsers.add_parser("da", help="offline ensemble data assimilation (LETKF, experimental, d01)",
                               description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = da.add_subparsers(dest="da_command", metavar="<da-command>")
    check = sub.add_parser("check", help="validate a DA config and list every input it would read")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--json", action="store_true", help="include the full read plan")
    check.set_defaults(func=cmd_check)
    analyse = sub.add_parser("analyse", aliases=["analyze"], help="one offline LETKF analysis time")
    analyse.add_argument("--config", required=True, type=Path)
    analyse.add_argument("--out", required=True, type=Path, help="new output directory (must not exist)")
    analyse.set_defaults(func=cmd_analyse)
    try:   # the cycle lane mounts 'da cycle' (+ 'da members') when its module is present
        from .cycle_loop import register as register_cycle
    except ImportError:
        register_cycle = None
    if register_cycle is not None:
        register_cycle(sub)
    else:
        cycle = sub.add_parser("cycle", help="forecast/analysis cycles with the GPU member driver (not in this build)")
        cycle.add_argument("--config", type=Path)
        cycle.add_argument("--out", type=Path)
        cycle.set_defaults(func=cmd_cycle_unavailable)

    def _no_sub(args):
        da.print_help(sys.stderr)
        return 2
    da.set_defaults(func=_no_sub)
    return da


def main(argv=None):
    """``python -m gpuwrf.assimilation.letkf <da-command> ...`` (same as ``gpuwrf da ...``)."""
    parser = argparse.ArgumentParser(prog="python -m gpuwrf.assimilation.letkf")
    subparsers = parser.add_subparsers(dest="command")
    register(subparsers)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(["da", *argv])
    return int(args.func(args))
