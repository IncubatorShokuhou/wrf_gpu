"""Native acoustic ``dampmag`` slot: static default vs traced dampcoef (critic FIX-2, constfold composition).

WRF advance_w forms ``dampmag = dts*config_flags%dampcoef`` as a REAL product at run time
(module_small_step_em.F:1284). The traced operand gives exactly that product. The static slot
folds ``cfg.dt*cfg.dampcoef`` in float64 and rounds once, which is 1 REAL ulp away from WRF at
dts 18 s and 4.5 s unless w44-constfold's ``GPUWRF_WRF_REAL_CONSTFOLD`` REAL fold is active, in
which case static and traced slots must be byte-identical. The test reads the slot vector the
production ``_w_phase`` builds and stops before any kernel launch.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig
from gpuwrf.kernels import dyn_acoustic_fp32 as N
from gpuwrf.kernels import fused_vertical_implicit as vi

try:  # present once lane/w44-constfold is composed
    from gpuwrf.contracts.real_constfold import ENABLED as REAL_FOLD
except ImportError:
    REAL_FOLD = False

SLOT = vi._SLOTS.index("dampmag")
ONE_ULP = (18., 4.5)  # PROD d01 RK1 and 3 km substeps (critic FIX-2)
EQUAL = (13.5, 6., 2., 1.5)


class Stop(Exception):
    def __init__(self, slots):
        self.slots = slots


class Staged:
    """Only shapes and scalars are read before the launch, which is intercepted."""

    def __init__(self):
        self.theta = jnp.zeros((2, 3, 3), jnp.float32)
        self.cf1 = self.cf2 = self.cf3 = jnp.float32(0.)

    def __getattr__(self, name):
        return jnp.zeros((3, 3, 3), jnp.float32)


def dampmag(monkeypatch, dts, dampcoef, traced):
    def launch(kernel, args, *rest, **kwargs):
        raise Stop(args["s"])

    monkeypatch.setattr(N, "_call", launch)
    cfg = AcousticCoreConfig(dt=dts, dx=3000., dy=3000., damp_opt=3, dampcoef=.2, zdamp=5000.)

    def run(value):
        selected = cfg if value is None else replace(cfg, dampcoef=value, damping_traced=True)
        try:
            N._w_phase(Staged(), {}, selected, interpret=True)
        except Stop as got:
            return got.slots
        raise AssertionError("production slot construction was not reached")

    slots = jax.jit(run)(jnp.float32(dampcoef)) if traced else run(None)
    return np.asarray(slots, np.float32)[SLOT]


def ulps(a, b):
    a, b = np.float32(a).view(np.int32), np.float32(b).view(np.int32)
    return abs(int(a) - int(b))


@pytest.mark.parametrize("dts", ONE_ULP + EQUAL)
def test_traced_dampmag_is_the_wrf_real_product(monkeypatch, dts):
    wrf = np.float32(dts) * np.float32(.2)
    traced = dampmag(monkeypatch, dts, .2, traced=True)
    assert traced.tobytes() == wrf.tobytes(), (dts, traced, wrf)


@pytest.mark.parametrize("dts", ONE_ULP + EQUAL)
def test_static_default_slot_class(monkeypatch, dts):
    static = dampmag(monkeypatch, dts, .2, traced=False)
    traced = dampmag(monkeypatch, dts, .2, traced=True)
    if REAL_FOLD:
        assert static.tobytes() == traced.tobytes(), "constfold ON: static and traced slots must be byte-identical"
    else:
        assert static == np.float32(dts * .2)  # pre-existing float64 fold, rounded once
        assert ulps(static, traced) == (1 if dts in ONE_ULP else 0), (dts, static, traced)
