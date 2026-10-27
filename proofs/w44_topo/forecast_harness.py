"""Retained S0 harness, three nests, with an output-boundary net-SW observer.

Only CPU history callbacks are wrapped; no compiled production function or
State leaf changes. GSW is recovered as the difference of the held terrain-
adjusted down/up fluxes; it is not recomputed by a radiation solve.

CPU launch smoke (TOPO_LAUNCH_SMOKE=1): the REAL model hook and CLI run on CPU
through case load, cold init and the hours/root-dt alignment check; the run
stops at the first post-init call (nest prewarm) and records the real carries.
Only device admission differs from the GPU child.
"""
import json
import os
from datetime import timedelta
from pathlib import Path
import sys


def install_output_observer(output=None):
    import jax
    import numpy as np
    from gpuwrf.integration import nested_pipeline as pipeline
    output = Path(output) if output is not None else Path(sys.argv[sys.argv.index('--out') + 1]) / 'held_surface'
    output.mkdir(parents=True, exist_ok=True)
    original = pipeline._PerDomainWrfoutWriter.__call__
    def observe(self, name, step, carry):
        rad = getattr(carry, 'radiation_diagnostics', None)
        if rad is None:
            raise ValueError('forecast effect observer requires held radiation')
        down, up = (np.asarray(x) for x in jax.device_get(
            (rad.swnorm, rad.swup_topographic)))
        if not all(np.isfinite(x).all() for x in (down, up)):
            raise ValueError('nonfinite held surface radiation')
        net = down - up
        if not np.isfinite(net).all():
            raise ValueError('nonfinite held GSW')
        valid = self.run_start + timedelta(seconds=int(step) * float(self.dt_by_domain[name]))
        np.savez_compressed(output / f'{name}_{valid:%Y-%m-%d_%H:%M:%S}.npz',
                            SWDOWN_TERRAIN=down, SWUP_TERRAIN=up, GSW=net)
        return original(self, name, step, carry)
    pipeline._PerDomainWrfoutWriter.__call__ = observe


def install_init_stop(output=None):
    """Stop the real CPU run after init + run-length validation; record carries."""
    import time
    import jax
    import numpy as np
    from gpuwrf.integration import nested_pipeline as pipeline
    output = Path(output) if output is not None else Path(sys.argv[sys.argv.index('--out') + 1])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hours = float(sys.argv[sys.argv.index('--hours') + 1])

    def stop(tree, carries=None, **kwargs):
        finite = nonfinite = 0
        shadow = {}
        for name, carry in dict(carries or {}).items():
            for leaf in jax.tree_util.tree_leaves(carry):
                value = np.asarray(leaf)
                if np.issubdtype(value.dtype, np.floating):
                    ok = bool(np.isfinite(value).all())
                    finite += ok
                    nonfinite += not ok
            rad = getattr(carry, 'radiation_diagnostics', None)
            height = getattr(rad, 'shadow_height_m', None)
            shadow[str(name)] = None if height is None else dict(
                dtype=str(np.asarray(height).dtype), shape=list(np.asarray(height).shape),
                all_zero=bool(np.all(np.asarray(height) == 0)), finite=bool(np.isfinite(np.asarray(height)).all()))
        record = dict(status='CPU_LAUNCH_SMOKE_PASS' if finite > 0 and nonfinite == 0 else 'CPU_LAUNCH_SMOKE_FAIL',
                      stop_point='post-init nested_pipeline.maybe_prewarm_defused_nest (after hours/root-dt alignment)',
                      hours=hours, domains=sorted(shadow), finite_leaves=finite, nonfinite_leaves=nonfinite,
                      shadow_height=shadow, backend=jax.devices()[0].platform,
                      init_wall_s=time.perf_counter() - started, model_hook='real (no substitution)')
        (output / 'LAUNCH_SMOKE_MODEL.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(LAUNCH_SMOKE=record['status'], init_wall_s=record['init_wall_s'])), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if record['status'] == 'CPU_LAUNCH_SMOKE_PASS' else 1)

    pipeline.maybe_prewarm_defused_nest = stop


def main():
    source = Path(os.environ['BENCH_SRC_REPO']) / 'scripts/bench/s0_nested_harness.py'
    code = source.read_text()
    old = '"--max-dom", "2", "--hours", str(args.hours)'
    assert code.count(old) == 1
    code = code.replace(old, '"--max-dom", "3", "--emit-initial-history", "--hours", str(args.hours)')
    smoke = os.environ.get('TOPO_LAUNCH_SMOKE') == '1'
    if smoke:
        # Change ONLY device admission for CPU launch proof. The real S0
        # argument parser, CLI validation/pre-exec and model hook still run.
        assert os.environ['JAX_PLATFORMS'] == 'cpu'
        guard = 'if device.platform != "gpu":'
        assert code.count(guard) == 1
        code = code.replace(guard, 'if device.platform != "cpu":')
    needle = 'if __name__ == "__main__":'
    assert code.count(needle) == 1
    hooks = 'MEASURED_HOOKS.append(_topo_output_observer)\n\n'
    if smoke:
        hooks += 'MEASURED_HOOKS.append(_topo_init_stop)\n\n'
    code = code.replace(needle, hooks + needle)
    exec(compile(code, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__',
        '_topo_output_observer': install_output_observer, '_topo_init_stop': install_init_stop})


if __name__ == '__main__':
    main()
