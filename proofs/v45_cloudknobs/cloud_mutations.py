"""In-memory semantic interventions against the frozen candidate; no source writes."""
import inspect
import os


def pytest_configure(config):
    from gpuwrf.optimize import cloud_controls as cc, study, continuous, identifiability as ident
    from gpuwrf.ens import workflows
    name = os.environ['CLOUD_MUTANT']
    table = {
        'M1': (cc, 'parameter', 'if row is None or not row["admitted"]:', 'if row is None:'),
        'M2': (cc, 'expand', 'row["default"] * factor', 'row["default"]'),
        'M3': (cc, 'parameter', 'domains = item.get("domains", list(run_domains))', 'domains = list(run_domains)'),
        'M4': (study, 'run', 'kwargs["runtime_parameters"] = parameters', 'pass  # delete per-member runtime forwarding'),
        'M5': (workflows, 'prepare', 'if runtime_parameters is not None:', 'if False and runtime_parameters is not None:'),
        'M6': (continuous, '_parameters', 'if len(out) > 2:', 'if len(out) > 200:'),
        'M7': (ident, 'restrict', 'rows = np.asarray(theta, np.float64) @ np.asarray(report["observable_projector"], np.float64)', 'rows = np.asarray(theta, np.float64).copy()'),
        'M8': (continuous, 'driver', 'state.theta = identifiability.restrict(state.theta, sensitivity, params)', 'state.theta = state.theta'),
        'M9': (ident, 'refuse_native_pixels', 'if any(str(location).startswith("msg3:") for location in location_ids):', 'if False:'),
        'M10': (continuous, 'load_continuous', 'identifiability.refuse_native_pixels(location_ids)', 'pass'),
        'M11': (cc, 'cloud_arm', 'raise ValueError("MYNN cloud arms require active bl_pbl_physics=5")', 'pass'),
        'M12': (cc, 'cloud_arm', 'raise ValueError("cu_physics=11 forces icloud=1 in WRF; the requested cloud arm would be ignored")', 'pass'),
        'M14': (ident, 'refuse_native_pixels', 'raise ValueError("correlated MSG pixels are refused: use admitted superobservations; " + requirement)', 'if not capability: raise ValueError("correlated MSG pixels are refused: use admitted superobservations; " + requirement)'),
    }
    module, attr, old, new = table[name]
    fn = getattr(module, attr)
    code = inspect.getsource(fn)
    assert code.count(old) == (2 if name == 'M8' else 1), (name, code.count(old))
    namespace = dict(fn.__globals__)
    exec(compile(code.replace(old, new), f'<cloud-semantic-{name}>', 'exec'), namespace)
    setattr(module, attr, namespace[attr])
