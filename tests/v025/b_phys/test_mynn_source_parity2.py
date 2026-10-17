"""Two remaining MYNN source primitives against original REAL4 WRF."""
import ctypes
import hashlib
import inspect
from pathlib import Path
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import mynn_edmf as E, mynn_pbl as P

KEY = "GPUWRF_MYNN_SOURCE_PARITY2"
HERE = Path(__file__).parent
COHORTS = ("sea_sc12", "land_night00")


def finite(*arrays):
    for array in arrays:
        assert np.isfinite(np.asarray(array)).all(), "non-finite oracle or port operand/result"


@pytest.fixture(autouse=True)
def enable(monkeypatch):
    monkeypatch.setenv(KEY, "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    phys = HERE.parents[2] / "data/wrf_pristine/WRF/phys"
    source = phys / "module_bl_mynnedmf.F"
    text = source.read_text()
    body = re.search(r"!stability criteria for mf\s+(DO k=kts\+1,kte-1\s+khdz\(k\).*?ENDDO)", text, re.S | re.I).group(1)
    code = """subroutine floor(n,km,kh,aw,om,oh) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),intent(in)::km(n+1),kh(n+1),aw(n+1)
real(c_float),intent(out)::om(n+1),oh(n+1)
real(c_float)::kmdz(n+1),khdz(n+1),s_aw1(n+1),sd_aw1(n+1)
integer::k,kts,kte
kts=1;kte=n;kmdz=km;khdz=kh;s_aw1=aw;sd_aw1=0.
""" + body + "\nom=kmdz;oh=khdz\nend subroutine\n"
    code += """subroutine original_gtr(out) bind(C)
use iso_c_binding
use module_bl_mynnedmf_common,only:gtr
implicit none
real(c_float),intent(out)::out
out=gtr
end subroutine
"""
    root = tmp_path_factory.mktemp("original_parity2")
    (root / "original.f90").write_text(code)
    subprocess.run(["<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran", "-O2", "-ffp-contract=off", "-ffree-line-length-none", "-fPIC", "-shared", "-I" + str(phys), "original.f90", "-o", "original.so"], cwd=root, check=True, capture_output=True, timeout=60)
    (root / "binding.txt").write_text(str(source) + "\n" + hashlib.sha256(source.read_bytes()).hexdigest() + "\n" + body)
    lib = ctypes.CDLL(str(root / "original.so"))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.floor.argtypes = [ctypes.c_int] + [ptr] * 5
    lib.original_gtr.argtypes = [ptr]
    return lib


def load(label):
    with np.load(HERE / ("fixtures/mynn_parity2_" + label + ".npz"), allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    finite(*values.values())
    assert len(values["km"]) == (518 if label == "sea_sc12" else 504)
    return values


@pytest.mark.parametrize("label", COHORTS)
def test_original_mean_floor_all_real_columns(label, original):
    data = load(label)
    km, kh, aw = (data[name] for name in ("km", "kh", "s_aw"))
    om, oh = np.empty_like(km), np.empty_like(kh)
    for i in range(len(km)):
        original.floor(km.shape[-1] - 1, km[i], kh[i], aw[i], om[i], oh[i])
    actual_m = np.asarray(P._apply_s_aw_stability_floor(jnp.asarray(km), jnp.asarray(aw)))
    actual_h = np.asarray(P._apply_s_aw_stability_floor(jnp.asarray(kh), jnp.asarray(aw)))
    finite(om, oh, actual_m, actual_h)
    np.testing.assert_array_equal(actual_m.view(np.uint32), om.view(np.uint32))
    np.testing.assert_array_equal(actual_h.view(np.uint32), oh.view(np.uint32))


def test_original_top_face_footprint_witness(original):
    # Activity witness only: put an existing real MF value at the top face.
    # This verifies the literal loop footprint, not a forecast effect claim.
    data = load("sea_sc12")
    km, kh, aw = (data[name][0].copy() for name in ("km", "kh", "s_aw"))
    aw[-2] = max(np.max(data["s_aw"]), np.float32(4) * max(km[-2], kh[-2]))
    om, oh = np.empty_like(km), np.empty_like(kh)
    original.floor(len(km) - 1, km, kh, aw, om, oh)
    actual = np.asarray(P._apply_s_aw_stability_floor(jnp.asarray(kh), jnp.asarray(aw)))
    finite(km, kh, aw, om, oh, actual)
    np.testing.assert_array_equal(actual.view(np.uint32), oh.view(np.uint32))
    assert actual[-2] == kh[-2]


@pytest.mark.parametrize("label", COHORTS)
def test_original_dmp_gtr_all_real_columns(label, original):
    data = load(label)
    ref = np.empty((), np.float32)
    original.original_gtr(ref)
    # The coefficient consumed once per call, independent of activation.
    actual = np.asarray(jax.vmap(E._dmp_gtr)(jnp.asarray(data["fltv"])))
    finite(ref, actual)
    np.testing.assert_array_equal(actual.view(np.uint32), np.full(actual.shape, ref.view(np.uint32), np.uint32))
    assert int(ref.view(np.uint32)) == 1023799408


def test_actual_dmp_setup_consumes_gtr(monkeypatch):
    data = load("sea_sc12")
    seen = []
    class Stop(Exception):
        pass
    def spy(like):
        seen.append(np.asarray(like))
        raise Stop
    monkeypatch.setattr(E, "_dmp_gtr", spy)
    names = list(inspect.signature(E._dmp_setup).parameters)
    args = {name: jnp.asarray(data[name][0]) for name in names if name in data}
    with pytest.raises(Stop):
        E._dmp_setup(**args, dx=3000.)
    assert len(seen) == 1
    finite(*seen)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_off_and_double_unchanged(monkeypatch, dtype):
    with jax.enable_x64(True):
        data = load("sea_sc12")
        kd, aw = jnp.asarray(data["kh"], dtype), jnp.asarray(data["s_aw"], dtype)
        monkeypatch.setenv(KEY, "0")
        before = np.asarray(P._apply_s_aw_stability_floor(kd, aw))
        expected = np.concatenate((np.asarray(kd)[..., :1], np.maximum(np.maximum(np.asarray(kd)[..., 1:-1], .5 * np.asarray(aw)[..., 1:-1]), -.5 * (np.asarray(aw)[..., 1:-1] - np.asarray(aw)[..., 2:])), np.asarray(kd)[..., -1:]), axis=-1)
        finite(before, expected)
        assert before.tobytes() == expected.tobytes()
        assert E._dmp_gtr(jnp.asarray(1, dtype)) == E.GTR
        if dtype == jnp.float64:
            monkeypatch.setenv(KEY, "1")
            assert np.asarray(P._apply_s_aw_stability_floor(kd, aw)).tobytes() == before.tobytes()
            assert E._dmp_gtr(jnp.asarray(1, dtype)) == E.GTR


@pytest.mark.parametrize("poison", (np.nan, np.inf, -np.inf))
def test_finite_gate_rejects_poison(poison):
    with pytest.raises(AssertionError, match="non-finite"):
        finite(np.asarray(poison, np.float32))
