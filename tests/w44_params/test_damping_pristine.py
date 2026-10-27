"""Changed damping controls against untouched WRF advance_w on real-derived inputs.

The production caller constructs an acoustic stage from the real case input.
An interior 4x4 crop retains every vertical level. It is an operator witness,
not a cropped forecast or a comparison against another gpuwrf version.
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import dyn_acoustic_fp32 as N
from gpuwrf.runtime import operational_mode as O

ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_FORCING = ("u_work_bdy", "v_work_bdy", "ph_bdy_target", "w_spec_target",
                    "mu_spec_target", "u_spec_tan_target", "v_spec_tan_target")


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage():
    helper = load("params_damping_real_inputs", "tests/v025/b_core/prod_inputs.py")
    _, bundles, _, _, _, carries = helper.prod_domains(root_only=True)
    nl = replace(bundles["d01"].namelist, run_physics=False)
    cfg_seen = []

    class Ready(Exception):
        def __init__(self, state):
            self.state = state

    def coefficients(state, cfg, **kwargs):
        cfg_seen.append(cfg)
        nz, ny, nx = state.theta.shape
        assert ny > 34 and nx > 54

        def crop(value):
            if value is None or value.ndim < 2:
                return value
            assert value.shape[-2] in (ny, ny + 1) and value.shape[-1] in (nx, nx + 1)
            return value[..., 30:34 + (value.shape[-2] - ny), 50:54 + (value.shape[-1] - nx)]

        cropped = jax.tree.map(crop, state)
        # The interior crop has no lateral boundary: WRF advance_w never reads
        # the specified-boundary forcing, which the boundary pass owns.
        raise Ready(cropped.replace(**{name: None for name in BOUNDARY_FORCING}))

    def prepare(carry):
        try:
            O._physics_boundary_step(carry, nl, jnp.int32(0), run_radiation=False)
        except Ready as result:
            return result.state
        raise AssertionError("production acoustic staging was not reached")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GPUWRF_DYN_FP32", "1")
        patch.setattr(N, "calc_coef_fp32", coefficients)
        state = jax.jit(prepare)(carries["d01"])
        jax.block_until_ready(state)
    assert len(cfg_seen) == 1
    assert cfg_seen[0].damp_opt == 3
    # Crop edges are not domain edges: the specified-W projection belongs to
    # the boundary pass; the pristine oracle receives the same specified flag.
    return state, replace(cfg_seen[0], spec_w_zero_grad=False)


def gate(actual, reference):
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    delta = actual - reference
    # Existing b_core/bench_acoustic.py compare() limits, unchanged.
    assert np.sqrt(np.mean(delta ** 2)) <= 2e-5 * max(1., np.sqrt(np.mean(reference ** 2)))
    assert np.max(np.abs(delta)) <= 2e-4 * max(1., np.max(np.abs(reference)))


def difference(actual, reference):
    a, b = np.asarray(actual, np.float32), np.asarray(reference, np.float32)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ordered = lambda value: np.where(value.view(np.uint32) >> 31,
        ~value.view(np.uint32), value.view(np.uint32) | np.uint32(0x80000000)).astype(np.int64)
    return dict(max_abs=float(error.max()),
                max_rel_nonzero_reference=float(np.max(error[b != 0] / np.abs(b[b != 0]), initial=0)),
                changed_cells=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
                max_ulp=int(np.max(np.abs(ordered(a) - ordered(b)))))


def test_changed_dampcoef_and_zdamp_keep_original_wrf_bounds(stage, tmp_path):
    oracle = load("params_damping_original_wrf", "tests/v025/b_core/pristine_oracle.py")
    library = oracle.build(tmp_path / "original")
    state, cfg = stage
    coefficients = jax.jit(lambda s: N.calc_coef_fp32(s, cfg, interpret=True))(state)
    traces = []

    def run(state, coef, controls):
        traces.append(1)
        selected = cfg if controls is None else replace(cfg,
            dampcoef=controls[0], zdamp=controls[1], damping_traced=True)
        result = N._w_phase(state, coef, selected, interpret=True)
        return result.w, result.ph, result.t_2ave

    call = jax.jit(run)
    static = call(state, coefficients, None)
    reports, references, outputs = [], [], []
    for label, dampcoef, zdamp in (("default", .2, 5000.), ("dampcoef", .4, 5000.), ("zdamp", .2, 7000.)):
        controls = jnp.asarray([dampcoef, zdamp], jnp.float32)
        actual = call(state, coefficients, controls)
        original = oracle.Oracle(library, state, replace(cfg, dampcoef=dampcoef, zdamp=zdamp))
        original.run(1)  # original calc_coef_w supplies the original solver coefficients
        original.run(4)  # unchanged advance_w, including WRF REAL damping product
        truth = tuple(original.field(name) for name in ("w", "ph", "t_2ave"))
        for got, wanted in zip(actual, truth, strict=True):
            gate(got, wanted)
        reports.append(dict(label=label, dampcoef=dampcoef, zdamp=zdamp,
                            differences_from_static=[difference(a, b) for a, b in zip(actual, static, strict=True)]))
        references.append(truth)
        outputs.append(tuple(np.asarray(a) for a in actual))
    for label, index in (("dampcoef", 1), ("zdamp", 2)):
        assert np.any(references[index][0] != references[0][0]), label + " inactive in original WRF"
        assert np.any(outputs[index][0] != outputs[0][0]), label + " did not reach native damping"
    assert traces == [1, 1], "values must share one ON trace"
    print("DAMPING_ORACLE", json.dumps(dict(dt=cfg.dt, shape=list(state.theta.shape), fields=["w", "ph", "t_2ave"], rows=reports)))
    call.clear_cache()
