"""pytest plugin: apply ONE source mutant (env W41_MUTANT) to a function before the tests run (lane w41-da-ens).
Each mutant replaces an exact source fragment of one function (must occur once) and re-executes the function in
its own module namespace, so every caller sees the mutated code."""
import importlib
import inspect
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # proofs/a1_da/ens (osse_tables)

MUTANTS = {
    # P1: the state inflation increment does not reach P/PH (the pre-fix behaviour)
    "M1_no_rebalance": ("gpuwrf.assimilation.letkf.inflation", "inflate_state",
                        "    if not rebalance:\n", "    if True:\n"),
    # P2: relative inflation paths pass through unresolved (the pre-fix behaviour)
    "M2_paths_passthrough": ("gpuwrf.assimilation.letkf.inflation", "_opt_path",
                             "    path = Path(value).expanduser()\n", "    return value\n"),
    # P3: per-band divergence not band-selected
    "M3_band_mix": ("gpuwrf.assimilation.letkf.perturb", "balance_diagnostics",
                    "d = div[..., 1:-1, 1:-1][..., sel_d]", "d = div[..., 1:-1, 1:-1]"),
    # v2 estimator: coherent forecast-error variance dropped from the ObsBias gain
    "M4_no_coherent": ("gpuwrf.assimilation.letkf.inflation", "ObsBias.update",
                       "if self.coherent else 0.0", "if False else 0.0"),
    # v2 estimator: levels counted as independent samples again
    "M5_levels_as_samples": ("gpuwrf.assimilation.letkf.inflation", "AdaptiveInflation.update",
                             "n_eff = m_eff if sid is None else len(set(sid[sel])) * m_eff / max(n, 1)",
                             "n_eff = m_eff"),
    # OSSE gate: completeness check disabled (a report with only the 03Z row passed vacuously)
    "M6_gate_no_completeness": ("osse_tables", "completeness", "    return why\n", "    return []\n"),
    # public config: the config directory is not forwarded to the inflation validator
    "M7_config_no_base": ("gpuwrf.assimilation.letkf.config", "_delegated",
                          "if base is not None and ", "if False and "),
    # OSSE gate: registered valid times not checked
    "M8_gate_no_time_check": ("osse_tables", "completeness", 'if r.get("t_a") != EXPECTED_T_A.get(r.get("cycle")):',
                              "if False:"),
    # OSSE gate: G2 baseline = first row as given (pre-fix), not the cycle-0 prior
    "M9_gate_first_row_base": ("osse_tables", "gate", 'base = next(r for r in rows if r["cycle"] == 0)["prior_spread"]',
                               'base = a1["rows"][0]["prior_spread"]'),
    # case boundary width ignored (config/default 5) in the adaptive state inflation
    "M10_inflation_default_width": ("gpuwrf.assimilation.letkf.inflation", "inflate_state",
                                    "resolve_spec_bdy_width(geo or {}, spec_bdy_width)",
                                    "resolve_spec_bdy_width({}, spec_bdy_width)"),
    # case boundary width ignored in the perturbed-IC writer
    "M11_ic_default_width": ("gpuwrf.assimilation.letkf.ensemble", "perturbed_ic_members",
                             "resolve_spec_bdy_width(geo, spec_bdy_width)", "resolve_spec_bdy_width({}, spec_bdy_width)"),
}


def pytest_configure(config):
    name = os.environ.get("W41_MUTANT")
    if not name:
        return
    module_name, qual, old, new = MUTANTS[name]
    module = importlib.import_module(module_name)
    owner, attr = (module, qual) if "." not in qual else (getattr(module, qual.split(".")[0]), qual.split(".")[1])
    func = getattr(owner, attr)
    src = textwrap.dedent(inspect.getsource(func))
    if src.count(old) != 1:
        raise RuntimeError(f"{name}: fragment occurs {src.count(old)} times")
    ns = dict(module.__dict__)
    exec(compile(src.replace(old, new), f"<mutant {name}>", "exec"), ns)
    setattr(owner, attr, ns[attr])
    config.w41_mutant = name
