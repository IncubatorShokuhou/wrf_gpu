"""t-calleraudit: the RELEASE MYNN path lowers with WRF's passive number scalars (FINAL c18-c24 blocker).

F8 (dac4a66631) hands MYNN's DMP_mf the MP's number scalars beyond QNI (QNC; QNWFA/QNIFA for mp 28/38)
as a dict ``passive``. Only the retained ``mynn_edmf.dmp_mf_columns`` understood it; the release entry
(GPUWRF_MYNN_FP32_COLUMNS -> phys_mynn_columns._native_edmf_arrays -> phys_mynn_plume.dmp_mf_columns_native)
cast every keyword with jnp.asarray, so every bl5 composite whose MP carries Nc died at d01 lowering with
"TypeError: float() argument must be a string or a real number, not 'dict'" (v-prep FINAL-10/11 on
04cb09f816: c18 mp26, c19 mp28, c20 mp40, c22-c24 mp51/52/53; also c65 mp38, c88 mp56).

WRF DMP_mf (module_bl_mynnedmf.F :6099-6103, :6172-6176, :6411-6420) runs the QNI passive entrainment
recurrence and flux limiter for UPQNC/UPQNWFA/UPQNIFA, so the native entry evaluates each species on the
native QNI lane of an otherwise identical plume (the retained design of F8).
Oracle: F8's pristine DMP_mf fixture (16 real CPU-WRF c19 columns), QNI 5 % gate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests/v040/rootbdy"))

from test_t_cfix_pblmix_mynn_numbers import NAMES, W, _a, _mf_args  # noqa: E402

from gpuwrf.kernels import phys_mynn_plume as L  # noqa: E402

# The release MYNN dispatch (src/gpuwrf/_fast_defaults.py), read at call time by the column entry.
RELEASE_MYNN = (("GPUWRF_MYNN_FP32_COLUMNS", "1"), ("GPUWRF_MYNN_PLUME_SUMS", "1"),
                ("GPUWRF_EDMF_FUSED_PLUME", "1"), ("GPUWRF_MYNN_FP32_PLUME", "1"))


@pytest.fixture(autouse=True)
def clear_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def test_native_entry_passive_fluxes_match_pristine_dmp_mf(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_PLUME_SUMS", "1")
    args, kw = _mf_args(jnp.float32)
    passive = kw.pop("passive")

    def run(**extra):
        return jax.jit(lambda *a: L.dmp_mf_columns_native(*a, interpret=True, **kw, **extra))(*args)

    joint, base = run(passive=passive), run()
    for name, (_, _, flux) in NAMES.items():
        got, ref = np.asarray(joint["s_awq" + name], np.float64), _a(W, flux, np.float64)
        assert np.isfinite(got).all(), name
        for c in range(len(ref)):
            scale = np.abs(ref[c]).max()
            assert np.abs(got[c] - ref[c]).max() <= (0.05 * scale if scale else 1e-12), (name, c)
        # each species IS the native QNI lane of an identical evaluation (bit for bit)
        alone = jax.jit(lambda *a, q=passive[name]: L.dmp_mf_columns_native(
            *a, interpret=True, **dict(kw, qni=q)))(*args)
        np.testing.assert_array_equal(np.asarray(joint["s_awq" + name]), np.asarray(alone["s_awqni"]))
    assert any(np.abs(_a(W, flux)).max() > 0 for _, _, flux in NAMES.values())  # oracle not vacuous
    # passive lanes do not feed back: every output of the plain call is unchanged
    assert set(base) <= set(joint)
    for key in base:
        np.testing.assert_array_equal(np.asarray(joint[key]), np.asarray(base[key]), err_msg=key)


def _lower_root_step(mp: int, tmp_path: Path) -> dict:
    """Lower the real Swiss root d01 step (the FINAL composites' geometry) with ``mp`` on the release MYNN
    path and record what the native plume entry received/returned. Runs in a FRESH process (see below)."""
    from _pytest.monkeypatch import MonkeyPatch
    from test_root_flow_extras import _swiss_root
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_mode import _pbl_mynn_number_scalars
    patch = MonkeyPatch()
    try:
        for key, value in RELEASE_MYNN:
            patch.setenv(key, value)
        nml, carry = _swiss_root(mp, tmp_path, patch)
        seen = []
        real = L.dmp_mf_columns_native

        def spy(*a, **k):
            out = real(*a, **k)
            seen.append([sorted(k.get("passive") or ()), sorted(out)])
            return out
        patch.setattr(L, "dmp_mf_columns_native", spy)
        shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
        lowered = op._advance_chunk_fori.lower(
            shapes, nml, jnp.asarray(1, jnp.int32), op.build_clock_base(nml), n_steps=1,
            cadence=int(nml.radiation_cadence_steps))
        # lowered for real; no module text (that alone costs ~1.8 GB RSS on the Swiss step)
        return dict(mp=int(nml.mp_physics), bl=int(nml.bl_pbl_physics), names=list(_pbl_mynn_number_scalars(mp)),
                    out_matches_carry=jax.tree.structure(lowered.out_info) == jax.tree.structure(shapes),
                    platform=jax.devices()[0].platform, seen=seen)
    finally:
        patch.undo()


@pytest.mark.parametrize("mp", (26, 28, 40, 51, 52, 53, 38, 56))
def test_release_mynn_root_step_lowers_with_number_scalars(tmp_path, mp):
    """The d01 step program LOWERS on CPU for every FINAL bl5 x Nc-MP composite's MP, and the native plume
    really received and returned the Registry number set. One fresh process per MP: a Swiss build + lower
    peaks ~4 GB and in-process traces accumulate past the 6 GB gate slot (measured OOM at 4/9 tests)."""
    import gpuwrf
    from gpuwrf.kernels.phys_mynn_numbers import mf_flux_key
    from test_root_flow_extras import SWISS
    if not (SWISS / "namelist.input").exists():
        pytest.skip(f"Swiss case unavailable: {SWISS}")
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0",   # E50/E68: never a GPU or shared cache
               PYTHONPATH=os.pathsep.join([str(Path(gpuwrf.__file__).resolve().parents[1]),
                                           os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, __file__, str(mp), str(tmp_path)], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-6000:]
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    names = rec["names"]
    assert rec["mp"] == mp and rec["bl"] == 5 and names and rec["platform"] == "cpu", rec
    assert rec["out_matches_carry"], rec
    assert rec["seen"], "release MYNN entry not dispatched (E114)"
    for got, keys in rec["seen"]:
        assert set(got) == set(names), (got, names)   # lanes are keyed by species (pytree-sorted dict)
        assert {mf_flux_key(n) for n in names} <= set(keys), keys


if __name__ == "__main__":
    print(json.dumps(_lower_root_step(int(sys.argv[1]), Path(sys.argv[2]))))
