#!/usr/bin/env python3
"""Check documented CLI examples, local links and WRF history names; no forecasts.

Run from an immutable checkout with JAX_PLATFORMS=cpu and GPUWRF_JAX_CACHE=0.
Optional final wave checks deliberately fail until public interfaces are present.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
PAGES = [
    "docs/regional_optimization.md", "docs/radiation_diagnostics.md",
    "docs/radiation_pbl_clouds.md", "docs/initial_overrides.md",
    "docs/noahmp_land_use.md", "docs/soil_porosity.md",
    "docs/checkpoint-branching.md", "docs/precompile.md",
]


def anchors(text):
    found = set(re.findall(r'\bid=["\']([^"\']+)', text))
    used = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+)$", text):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        n = used.get(slug, 0)
        used[slug] = n + 1
        found.add(slug + (f"-{n}" if n else ""))
    return found


def commands(text):
    for block in re.findall(r"(?ms)^\s*```bash\n(.*?)^\s*```", text):
        for line in block.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if "gpuwrf" in tokens:
                yield tokens[tokens.index("gpuwrf") + 1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=ROOT / "data/wrf_pristine/WRF/Registry/Registry.EM")
    ap.add_argument("--require-v044", action="store_true")
    ap.add_argument("--require-merged-tools", action="store_true",
                    help="check only lutab/override/raddiag/branch/W5; leave open lanes out")
    args = ap.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("JAX_PLATFORMS=cpu is required")
    sys.path.insert(0, str(ROOT / "src"))
    from gpuwrf import cli
    assert Path(cli.__file__).resolve().is_relative_to(ROOT), cli.__file__
    parser = cli.build_parser()
    # Abbreviations would conceal misspelled documented flags.
    def exact(p):
        p.allow_abbrev = False
        for a in p._actions:
            if isinstance(a, argparse._SubParsersAction):
                for child in a.choices.values():
                    exact(child)
    exact(parser)
    registry_sources = {}
    def read_registry(path):
        path = path.resolve()
        if str(path) in registry_sources:
            return ""
        raw = path.read_bytes()
        registry_sources[str(path)] = hashlib.sha256(raw).hexdigest()
        content = raw.decode()
        included = [read_registry(path.parent / name) for name in
                    re.findall(r"(?m)^include\s+(\S+)", content)]
        return content.replace("\\\n", " ") + "\n" + "\n".join(included)
    registry = read_registry(args.registry)
    registered = set()
    for line in registry.splitlines():
        if re.match(r"\s*state\s", line, re.I):
            quoted = re.findall(r'"([^"]*)"', line)
            if quoted:
                # gen_allocs.c uppercases DataName; modify_io_masks is case-insensitive.
                registered.add(quoted[0].upper())
    failures, parsed, links, fields = [], [], [], set()
    hashes = {}
    for name in PAGES:
        path = ROOT / name
        text = path.read_text()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        prose = re.sub(r"\]\([^)]+\)", "]", prose)
        prose = re.sub(r"`[^`]+`", "", prose)
        internal = re.findall(r"\b(?:W4|W5|PROD|WN3|manager)\b", prose, re.I)
        if internal:
            failures.append({"page": name, "undefined_internal_labels": sorted(set(internal))})
        for argv in commands(text):
            error = io.StringIO()
            try:
                with contextlib.redirect_stdout(error), contextlib.redirect_stderr(error):
                    parser.parse_args(argv)
                parsed.append({"page": name, "argv": argv})
            except SystemExit as exc:
                failures.append({"page": name, "argv": argv, "error": error.getvalue(), "rc": exc.code})
        for url in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            parts = urlsplit(url.strip("<>"))
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            ok = target.exists()
            if ok and parts.fragment and target.suffix in {".md", ".html"}:
                ok = unquote(parts.fragment) in anchors(target.read_text())
            row = {"page": name, "url": url, "passed": ok}
            links.append(row)
            if not ok:
                failures.append(row)
        for names in re.findall(r"(?m)^\+:h:0:(.+)$", text):
            for field in names.split(","):
                fields.add(field.strip())
    for field in sorted(field for field in fields if field.upper() not in registered):
        failures.append({"history_field": field, "error": "absent from WRF Registry output names"})
    # Record the current catalog without claiming pending selections are supported.
    from gpuwrf.io import scheme_catalog as cat
    catalog = {}
    for key, value in [("slope_rad", 1), ("slope_rad", 2), ("topo_shading", 1),
                       ("icloud_bl", 0), ("bl_mynn_edmf", 0), ("tke_budget", 1), ("bl_mynn_output", 1)]:
        verdict = cat.classify_control(key, value) or cat.classify_feature_switch(key, value)
        catalog[f"{key}={value}"] = {
            "known": key in cat.OUT_OF_SCOPE_FEATURE_KEYS or key in cat.RECOGNIZED_CONTROL_KEYS
            or key in cat.IMPLEMENTED_CONTROL_KEYS,
            "refused": verdict is not None and verdict.status != cat.SupportStatus.IMPLEMENTED,
        }
    for key, expected in [("slope_rad=1", False), ("slope_rad=2", True), ("topo_shading=1", False)]:
        if catalog[key] != {"known": True, "refused": expected}:
            failures.append({"catalog_mismatch": key, **catalog[key]})
    merged_tools = {}
    if args.require_merged_tools:
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
        help_checks = {
            "override": ("--spec", "--surface-physics", "--force-no-effect"),
            "precompile": ("--namelist-variants", "--execute", "--timeout-seconds"),
            "run": ("--branch-from", "--branch-manifest", "--minutes", "--end-time"),
        }
        for name, options in help_checks.items():
            help_text = sub[name].format_help()
            merged_tools[name] = {option: option in help_text for option in options}
            for option, found in merged_tools[name].items():
                if not found:
                    failures.append({"missing_merged_help": name, "option": option})
        if catalog["icloud_bl=0"] != {"known": True, "refused": False}:
            failures.append({"merged_cloud_switch": catalog["icloud_bl=0"]})
        from gpuwrf.io.namelist_binding import bound_options, namelist_honour_report
        raw = {"gpuwrf": {"rad_diag": [0, 1, 1]}, "physics": {
            "ra_lw_physics": [4, 4, 4], "ra_sw_physics": [4, 4, 4],
            "icloud_bl": 0, "mp_physics": [8, 8, 8], "cu_physics": [0, 0, 0]}}
        errors, _ = namelist_honour_report(raw, ("d01", "d02", "d03"))
        binding = {d: {k: bound_options(raw, d).get(k) for k in ("rad_diag", "icloud_bl")}
                   for d in ("d01", "d02", "d03")}
        merged_tools["radiation_namelist_binding"] = binding
        expected = {"d01": {"rad_diag": 0, "icloud_bl": 0},
                    "d02": {"rad_diag": 1, "icloud_bl": 0},
                    "d03": {"rad_diag": 1, "icloud_bl": 0}}
        if errors or binding != expected:
            failures.append({"radiation_binding": binding, "errors": errors})
        raw["physics"]["ra_lw_physics"] = [4, 1, 4]
        errors, _ = namelist_honour_report(raw, ("d02",))
        if not any("RRTMG" in e for e in errors):
            failures.append({"radiation_non_rrtmg_refusal_missing": errors})
        for lane, packet in [("lutab", "LUTAB"), ("override", "OVERRIDE"),
                             ("raddiag", "RADDIAG"), ("branch", "BRANCH"),
                             ("precompile", "PRECOMPILE")]:
            path = ROOT / f"proofs/w44_{lane}/{packet}.md"
            if not path.exists():
                failures.append({"missing_merged_packet": str(path)})
            else:
                merged_tools[f"{lane}_packet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.require_v044:
        for key in ("icloud_bl=0", "bl_mynn_edmf=0", "tke_budget=1", "bl_mynn_output=1"):
            if catalog[key] != {"known": True, "refused": False}:
                failures.append({"pending_v044_catalog": key, **catalog[key]})
        # Real parser existence, not a grep hit in prose.
        for argv in (["override", "--help"], ["cycle", "--help"]):
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    parser.parse_args(argv)
            except SystemExit as exc:
                if exc.code:
                    failures.append({"pending_v044_cli": argv, "rc": exc.code})
            if argv[0] == "cycle" and not re.search(r"atmos", output.getvalue(), re.I):
                failures.append({"pending_v044_cli": "cycle atmospheric selection"})
        for packet in ("raddiag/RADDIAG", "budget/BUDGET", "warmstart/WARMSTART",
                       "override/OVERRIDE", "topo/TOPO", "lutab/LUTAB"):
            lane, filename = packet.split("/")
            if not (ROOT / f"proofs/w44_{lane}/{filename}.md").exists():
                failures.append({"pending_v044_packet": packet})
    report = {
        "status": "FAIL" if failures else "PASS", "root": str(ROOT),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "docs/parser/Registry only; no model execution or numerical qualification",
        "require_v044": args.require_v044, "page_sha256": hashes,
        "require_merged_tools": args.require_merged_tools, "merged_tools": merged_tools,
        "registry_sources_sha256": registry_sources,
        "parsed_commands": parsed, "links": links, "cpu_history_names": sorted(fields),
        "catalog": catalog, "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "commit", "require_v044", "failures")}))
    print(f"{len(parsed)} commands; {len(links)} links; {len(fields)} WRF history names")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
