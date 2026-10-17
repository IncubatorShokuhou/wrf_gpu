"""Validate a stream file before launching a forecast (no JAX imports)."""
import json

from .config import read_streams


def command(args):
    try:
        specs = read_streams(args.input)
        print(json.dumps({"version": 1, "streams": [
            {"name": s.name, "domain": s.domain, "channels": s.channels,
             "interval_steps": s.steps(args.dt) if args.dt is not None else s.interval_steps,
             "interval_seconds": s.interval_seconds, "buffer_frames": s.buffer_frames}
            for s in specs]}, indent=2))
    except (ValueError, OSError) as error:
        print(f"gpuwrf: error: {error}")
        return 2
    return 0
