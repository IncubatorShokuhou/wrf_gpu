"""``gpuwrf aot-key``: per-program AOT cheap keys of a nested case, CPU only, no compile.

Loads the case like ``gpuwrf run`` (nested_pipeline._load_domains) in a CPU process and
prints the cheap key every step program would look up (aot_key_probe). With
``--compare-input-dir`` it loads a second case and reports, per program, whether both
cases share the program and which key component differs. A cache primed on one case
serves another case exactly when their programs match here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_COMPONENTS = ("program_config", "fn_identity", "source_fingerprint", "static_config", "carry_aval",
               "global_trace_env", "module_const_env", "baked_tables")


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "aot-key",
        help="print the AOT cheap keys a nested case would use (CPU only, no GPU, no compile)",
        description=(
            "Load a nested case on the CPU and print, per step program (root advance and "
            "fused cascades), the AOT cheap key it would look up. With --compare-input-dir, "
            "report which programs two cases share: a cache primed on one case serves the "
            "other exactly for the shared programs. Keys from this CPU process carry the CPU "
            "target, so compare cases with each other, not with a GPU cache listing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="case directory (as for gpuwrf run)")
    parser.add_argument("--namelist", type=Path, help="namelist.input to use instead of the one in --input-dir")
    parser.add_argument("--compare-input-dir", type=Path, help="second case directory to compare with")
    parser.add_argument("--compare-namelist", type=Path, help="namelist for --compare-input-dir")
    parser.add_argument("--hours", type=float, default=3.0, help="forecast length the keys are computed for")
    parser.add_argument("--json", type=Path, help="also write the full report (incl. components) as JSON")
    parser.set_defaults(func=command)


def _case_dir(input_dir: Path, namelist: Path | None, scratch: Path) -> Path:
    """``input_dir`` itself, or a symlink view of it with ``namelist`` as namelist.input."""
    if namelist is None:
        return input_dir
    view = Path(tempfile.mkdtemp(prefix="case_", dir=scratch))
    for item in input_dir.iterdir():
        if item.name != "namelist.input":
            (view / item.name).symlink_to(item.resolve())
    (view / "namelist.input").symlink_to(namelist.resolve())
    return view


def case_keys(input_dir: Path, *, hours: float, scratch: Path) -> dict[str, Any]:
    """Load one case and return its per-program key records (with components)."""
    from gpuwrf.cli import _namelist_max_dom
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.aot_key_probe import case_program_keys
    from gpuwrf.runtime.domain_tree import DomainTree

    max_dom = _namelist_max_dom(input_dir / "namelist.input")
    if max_dom < 2:
        raise ValueError("aot-key covers nested runs (max_dom >= 2); single-domain runs use another path")
    names = pipeline.domain_names_for(max_dom)
    config = pipeline.NestedPipelineConfig(input_dir, scratch / "out", scratch / "proof", hours=hours,
                                           max_dom=max_dom, scratch_dir=scratch)
    hierarchy, bundles, meta, _start, _dt, carries = pipeline._load_domains(config, names)
    nesting = meta.get("nesting", {"feedback": bool(config.feedback), "spec_zone": 1, "smooth_option": 1})
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=nesting["feedback"],
                                   feedback_spec_zone=nesting["spec_zone"],
                                   feedback_smooth_option=nesting["smooth_option"])
    return case_program_keys(tree, carries, components=True)


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows[name] = {"shared": False, "reason": "program only in one case"}
            continue
        if a[name].get("cheap_key") is None or b[name].get("cheap_key") is None:
            # A failed key computation proves nothing about the executables.
            rows[name] = {"shared": False, "unavailable": True, "reason": "no cheap key in "
                          + " and ".join(label for label, rec in (("case", a[name]), ("compare", b[name]))
                                         if rec.get("cheap_key") is None)}
            continue
        ca, cb = a[name].get("components", {}), b[name].get("components", {})
        rows[name] = {"shared": a[name]["cheap_key"] == b[name]["cheap_key"],
                      "differing_components": [c for c in _COMPONENTS if ca.get(c) != cb.get(c)]}
    return rows


def command(args: argparse.Namespace) -> int:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    try:  # the parser may already have imported jax; its backend is chosen on first use
        jax.config.update("jax_platforms", "cpu")
    except Exception:  # noqa: BLE001 - checked below
        pass
    if jax.default_backend() != "cpu":
        print("gpuwrf aot-key: error: JAX already runs on a non-CPU backend in this process", file=sys.stderr)
        return 2
    report: dict[str, Any] = {"schema": "gpuwrf.aot_key.v1", "platform": "cpu", "hours": args.hours,
                              "env_nested_aot": os.environ.get("GPUWRF_NESTED_AOT")}
    if os.environ.get("GPUWRF_NESTED_AOT", "1").strip().lower() in ("0", "false", "off", "no"):
        print("gpuwrf aot-key: note: GPUWRF_NESTED_AOT is off here, so a run in this environment loads no "
              "AOT executables; the keys below are those an AOT run would use", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="gpuwrf_aot_key_") as tmp:
        scratch = Path(tmp)
        cases = [("case", args.input_dir, args.namelist)]
        if args.compare_input_dir is not None:
            cases.append(("compare", args.compare_input_dir, args.compare_namelist))
        for label, input_dir, namelist in cases:
            case_scratch = scratch / label
            case_scratch.mkdir()
            directory = _case_dir(input_dir, namelist, case_scratch)
            report[label] = {"input_dir": str(input_dir), "namelist": str(namelist or input_dir / "namelist.input"),
                             "programs": case_keys(directory, hours=args.hours, scratch=case_scratch)}
    for label in ("case", "compare"):
        if label in report:
            print(f"{label}: {report[label]['input_dir']}")
            for name, record in report[label]["programs"].items():
                if record.get("cheap_key") is None:
                    print(f"  {name:12s} UNAVAILABLE ({record.get('unavailable', 'no cheap key')})")
                else:
                    print(f"  {name:12s} cheap_key={record['cheap_key']}")
    complete = all(record.get("cheap_key") is not None
                   for label in ("case", "compare") if label in report
                   for record in report[label]["programs"].values())
    if "compare" in report:
        report["comparison"] = _compare(report["case"]["programs"], report["compare"]["programs"])
        for name, row in report["comparison"].items():
            verdict = "UNAVAILABLE" if row.get("unavailable") else ("SHARED" if row["shared"] else "NOT SHARED")
            detail = row.get("reason") or ("" if row["shared"] else "differs in " + ", ".join(row["differing_components"]))
            print(f"  {name:12s} {verdict} {detail}".rstrip())
    report["complete"] = complete
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1, default=str) + "\n")
    if not complete:
        print("gpuwrf aot-key: error: some programs have no cheap key (see UNAVAILABLE)", file=sys.stderr)
        return 3
    return 0
