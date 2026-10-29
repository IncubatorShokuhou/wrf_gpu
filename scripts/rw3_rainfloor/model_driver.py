"""Backend assertion inside the measured model, after product allocator setup."""
import json
import os
from pathlib import Path
from gpuwrf import cli

assert Path(cli.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"]).resolve())
args = cli.build_parser().parse_args()
if args.dry_run:
    raise SystemExit(args.func(args))
cli._maybe_reexec_for_nested_allocator(args)
from gpuwrf.integration import nested_pipeline as pipeline
import jax

execute = pipeline._execute_nested_pipeline


def measured(*pos, **kw):
    device = jax.devices()[0]
    assert device.platform == "gpu", device
    Path("device_measured.json").write_text(json.dumps(dict(
        pid=os.getpid(), platform=device.platform, device=str(device), source_file=cli.__file__,
        scope="same process as measured model"), indent=2) + "\n")
    return execute(*pos, **kw)


pipeline._execute_nested_pipeline = measured
raise SystemExit(args.func(args))
