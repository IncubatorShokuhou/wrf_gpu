"""Deferred implementation of ``gpuwrf post``."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .io import read_wrfout, write_products
from .registry import PRODUCTS, load_application_products, required_inputs, run_products


def command(args):
    try:
        load_application_products()
        if args.list_products:
            print("\n".join(sorted(PRODUCTS)))
            return 0
        if not args.input or not args.output or not args.product:
            raise ValueError("--input, --output and at least one --product are required (or use --list-products)")
        source_paths = [Path(p).resolve() for p in args.input]
        if args.output.resolve() in source_paths:
            raise ValueError("output must differ from every input file")
        options = {}
        for item in args.option:
            key, sep, value = item.partition("=")
            if not sep or not key:
                raise ValueError("--option requires key=JSON (quote string values)")
            options[key] = json.loads(value)
        for key in ("heights", "rolling_hours", "bucket_mm", "storm_u", "storm_v"):
            value = getattr(args, key)
            if value is not None:
                options[key] = value
        names = [name for entry in args.product for name in entry.split(",")]
        ds = read_wrfout(args.input, required_inputs(names))
        result = run_products(ds, names, options)
        write_products(result, args.output)
    except (ValueError, OSError, TypeError, ImportError) as exc:
        print(f"gpuwrf post: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "products": names, "samples": result.sizes["Time"],
                      "variables": sorted(result.data_vars)}, sort_keys=True))
    return 0
