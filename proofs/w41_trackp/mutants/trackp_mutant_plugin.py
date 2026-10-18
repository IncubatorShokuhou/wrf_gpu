"""Pytest plugin: apply ONE Track P mutant (env TRACKP_MUTANT) by monkeypatching at configure time.

No source file is edited (E125). Each mutant must make at least one genuine assertion fail in tests/w41_trackp
(E202: setup errors do not count as kills).
"""
import os


def _apply(name):
    import numpy as np

    from gpuwrf.surface_params import controls as sc
    from gpuwrf.surface_params import eki

    if name == "M1_apply_drops_energy":
        real = sc.apply_surface_controls

        def mutant(static, ep, rp, controls):
            keep = {k: v for k, v in controls.items() if sc.CONTROLS[k].bundle != "energy"}
            return real(static, ep, rp, keep) if keep else (static, ep, rp)
        sc.apply_surface_controls = mutant
    elif name == "M2_phenology_ignores_lai_tables":
        from gpuwrf.physics.noahmp import noahmp_driver as drv
        real = drv.noahmp_phenology_table

        def mutant(*a, laim=None, saim=None, **k):
            return real(*a, **k)
        drv.noahmp_phenology_table = mutant
    elif name == "M3_restart_check_noop":
        sc.validate_controls_restart = lambda recorded, namelist: None
    elif name == "M4_aux_hashes_values":
        real = sc.signature
        sc.signature = lambda c: real(c) + (sc.controls_digest(c),)
    elif name == "M5_eki_innovation_sign":
        real = eki.update

        def mutant(problem, state, predictions, **k):
            new = real(problem, state, predictions, **k)
            return eki.EKIState(theta=2 * state.theta - new.theta, iteration=new.iteration, history=new.history)
        eki.update = mutant
    elif name == "M6_clock_base_drops_controls":
        from gpuwrf.runtime import operational_mode as om
        real = om.build_clock_base
        om.build_clock_base = lambda nl, *a, **k: real(nl, *a, **k)._replace(surface_controls=None)
    elif name == "M7_bounds_unchecked":
        real = sc.validate_controls

        def mutant(controls, namelist):
            saved = dict(sc.CONTROLS)
            sc.CONTROLS.update({k: v._replace(lower=-np.inf, upper=np.inf) for k, v in saved.items()})
            try:
                return real(controls, namelist)
            finally:
                sc.CONTROLS.clear()
                sc.CONTROLS.update(saved)
        sc.validate_controls = mutant
    elif name == "M8_expand_rounds_twice":
        real = sc.expand_controls

        def mutant(namelist, multipliers=None, *, values=None, device=True):
            out = dict(real(namelist, {k: 1.0 for k in (multipliers or {})}, device=False)) if multipliers else {}
            for k, m in (multipliers or {}).items():  # prior (already rounded) x multiplier, rounded again
                field = sc._multiplier_field(k, namelist, m)
                out[k] = (out[k].astype(np.float32) * field.astype(np.float32)).astype(out[k].dtype)
            if values:
                out.update(real(namelist, values=values, device=False))
            if device:
                import jax.numpy as jnp
                return {k: jnp.asarray(v) for k, v in out.items()}
            return out
        sc.expand_controls = mutant
    elif name == "M9_basis_preflight_noop":
        from gpuwrf.assimilation.letkf import member_driver
        member_driver.refuse_protected_inputs = lambda *a, **k: None
    elif name == "M12_basis_open_backstop_removed":
        import contextlib

        from gpuwrf.assimilation.letkf import member_driver
        member_driver.refuse_protected_opens = lambda: contextlib.nullcontext([])
    elif name == "M10_protected_ignores_symlinks":
        from pathlib import Path

        from gpuwrf.surface_params import stations as st

        def mutant(path):  # absolute() instead of resolve(): symlink targets no longer count
            p = Path(os.fsdecode(path)).absolute()
            return any(p == Path(r) or Path(r) in p.parents for r in st.ALISIOS_DATA_ROOTS)
        st.is_alisios_data = mutant
    elif name == "M11_session_covers_observations_only":
        from gpuwrf.surface_params import stations as st
        real = st._read_session
        st._read_session = lambda read_set, guard: real([read_set[1]], guard)  # the pre-FIX behaviour
    else:
        raise ValueError(f"unknown mutant {name}")


def pytest_configure(config):
    name = os.environ.get("TRACKP_MUTANT")
    if name:
        _apply(name)
