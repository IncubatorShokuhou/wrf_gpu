"""S0 three-nest profile plus complete old-carry byte hashes outside capture.

Only lead-zero and final (3h) output callbacks hash carry. The profiled second
hour has the ordinary production writer, so observer copies do not contaminate
the kernel/copy comparison. Nothing is added to a traced model function.
"""
import os
from pathlib import Path
import sys

source = Path(os.environ["BENCH_SRC_REPO"]) / "scripts/bench/s0_nested_harness.py"
code = source.read_text()
old = '"--max-dom", "2", "--hours", str(args.hours)'
assert code.count(old) == 1
code = code.replace(old, '"--max-dom", "3", "--emit-initial-history", "--hours", str(args.hours)')
namespace = {"__file__": str(source), "__name__": "raddiag_observed_profile"}
exec(compile(code, str(source), "exec"), namespace)


def observe_carry():
    import hashlib
    import json
    import jax
    import numpy as np
    from gpuwrf.integration import nested_pipeline as pipeline

    output = Path(sys.argv[sys.argv.index("--out") + 1]) / "CARRY_HASHES.json"
    original = pipeline._PerDomainWrfoutWriter.__call__
    records = {}

    def writer(self, name, own_step, carry, *args, **kwargs):
        result = original(self, name, own_step, carry, *args, **kwargs)
        lead = int(round(float(own_step) * self.dt_by_domain[name]))
        if lead not in (0, 10800):
            return result
        if not hasattr(carry, "radiation_diagnostics") or not hasattr(carry, "state"):
            raise ValueError("full-carry gate requires the real OperationalCarry output callback")
        radiation = carry.radiation_diagnostics
        normalized = carry.replace(radiation_diagnostics=radiation._replace(rad_diag=None))
        host = jax.device_get(normalized)
        leaves, _ = jax.tree_util.tree_flatten_with_path(host)
        values = {}
        for key, value in leaves:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"unhandled non-numeric carry leaf {key}")
            values[str(key)] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                   sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
        records[f"{name}:{lead}"] = dict(kind=type(carry).__name__, leaves=values,
                                       state_fields=list(carry.state.active_field_names()))
        output.write_text(json.dumps(dict(source=namespace["REC"].get("git_head"),
                                         omitted="only new rad_diag diagnostic volumes", records=records), indent=2)+"\n")
        return result

    pipeline._PerDomainWrfoutWriter.__call__ = writer


namespace["MEASURED_HOOKS"].append(observe_carry)
raise SystemExit(namespace["main"]())
