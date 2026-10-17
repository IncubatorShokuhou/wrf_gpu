"""Original WRF MYNN psi_init REAL tables and their actual lookup consumers."""
from pathlib import Path
import ctypes
import os
import re
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import surface_layer as S

KEY = "GPUWRF_MYNN_SFC_PSI_REAL"
NAMES = ("psim_stable", "psih_stable", "psim_unstable", "psih_unstable")


def finite(array):
    assert np.isfinite(array).all(), "non-finite oracle or port operand/result"
    return array


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    source = root / "data/wrf_pristine/WRF/phys/module_sf_mynn.F"
    assert source.is_file(), "original WRF source must be linked"
    text = source.read_text()
    init = re.search(r"SUBROUTINE psi_init\(psi_opt\).*?END SUBROUTINE psi_init", text, re.S | re.I).group(0)
    names = list(NAMES) + [n + "_full" for n in NAMES] + [n + "_full_gfs" for n in NAMES]
    blocks = [re.search(r"REAL function " + n + r"\(.*?end function", text, re.S | re.I).group(0) for n in names]
    code = "module original_psi\nreal :: psim_stab(0:1000),psih_stab(0:1000),psim_unstab(0:1000),psih_unstab(0:1000)\ncontains\n"
    code += init + "\n" + "\n".join(blocks) + "\nend module\n"
    code += """subroutine get_tables(out) bind(C)
use iso_c_binding
use original_psi
implicit none
real(c_float),intent(out)::out(1001,4)
integer::opt
opt=0
call psi_init(opt)
out(:,1)=psim_stab
out(:,2)=psih_stab
out(:,3)=psim_unstab
out(:,4)=psih_unstab
end subroutine
subroutine lookups(n,z,out) bind(C)
use iso_c_binding
use original_psi
implicit none
integer(c_int),value::n
real(c_float),intent(in)::z(n)
real(c_float),intent(out)::out(n,4)
integer::i,opt
opt=0
call psi_init(opt)
do i=1,n
out(i,1)=psim_stable(z(i),opt)
out(i,2)=psih_stable(z(i),opt)
out(i,3)=psim_unstable(-z(i),opt)
out(i,4)=psih_unstable(-z(i),opt)
enddo
end subroutine
subroutine stable_reciprocal(out) bind(C)
use iso_c_binding
implicit none
real(c_float),intent(out)::out
out=1./1.1
end subroutine
"""
    directory = tmp_path_factory.mktemp("original_psi")
    (directory / "original.f90").write_text(code)
    fc = shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    assert Path(fc).exists(), "original oracle needs gfortran"
    subprocess.run([fc, "-O2", "-ffp-contract=off", "-fPIC", "-shared", "original.f90", "-o", "original.so"], cwd=directory, check=True, capture_output=True, timeout=60)
    library = ctypes.CDLL(str(directory / "original.so"))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    library.get_tables.argtypes = [array]
    library.lookups.argtypes = [ctypes.c_int, array, array]
    library.stable_reciprocal.argtypes = [array]
    tables = np.empty((4, 1001), np.float32)
    library.get_tables(tables)
    finite(tables)
    return tables, library


@pytest.mark.parametrize("index", range(4))
def test_real_tables_bit_exact_original(original, monkeypatch, index):
    monkeypatch.setenv(KEY, "1")
    ref = finite(original[0][index])
    actual = finite(np.asarray(S._psi_table(jnp.asarray(0, jnp.float32), index)))
    np.testing.assert_array_equal(actual.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("index", range(4))
def test_actual_lookup_uses_original_tables(original, monkeypatch, index):
    monkeypatch.setenv(KEY, "1")
    # Every node plus half-node samples; avoid analytic endpoint >=10.
    z = np.concatenate((np.arange(1000, dtype=np.float32) * np.float32(.01),
                        (np.arange(1000, dtype=np.float32) + np.float32(.5)) * np.float32(.01)))
    finite(z)
    ref = np.empty((4, len(z)), np.float32)
    original[1].lookups(len(z), z, ref)
    finite(ref)
    argument = jnp.asarray(z if index < 2 else -z)
    actual = finite(np.asarray(getattr(S, "_" + NAMES[index])(argument)))
    # Non-jitted same-operation REAL interpolation: exact bytes.
    np.testing.assert_array_equal(actual.view(np.uint32), ref[index].view(np.uint32))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_off_and_double_tables_unchanged(monkeypatch, dtype):
    with jax.enable_x64(True):
        z = jnp.asarray(.4, dtype=dtype)
        monkeypatch.setenv(KEY, "0")
        off = [np.asarray(S._psi_table(z, i)).tobytes() for i in range(4)]
        expected = [np.asarray(t).tobytes() for t in S._TABLES[jnp.dtype(dtype)]]
        assert off == expected
        if dtype == jnp.float64:
            monkeypatch.setenv(KEY, "1")
            assert [np.asarray(S._psi_table(z, i)).tobytes() for i in range(4)] == off


def test_stable_reciprocal_original_real_bits(original, monkeypatch):
    monkeypatch.setenv(KEY, "1")
    ref = np.empty(1, np.float32)
    original[1].stable_reciprocal(ref)
    argument = jnp.asarray(12., jnp.float32)
    actual = np.asarray(S._psih_stable_reciprocal(argument))
    compiled = np.asarray(jax.jit(S._psih_stable_reciprocal)(argument))
    finite(ref); finite(actual); finite(compiled)
    assert int(ref.view(np.uint32)[0]) == 1063828014
    assert actual.tobytes() == compiled.tobytes() == ref.tobytes()


def test_actual_stable_fallback_consumes_real_reciprocal(original, monkeypatch):
    monkeypatch.setenv(KEY, "1")
    z = np.asarray([10., 10.001, 12., 20., 50., 100.], np.float32)
    ref = np.empty((4, len(z)), np.float32)
    original[1].lookups(len(z), z, ref)
    seen = []
    coefficient = S._psih_stable_reciprocal
    def spy(like):
        seen.append(np.asarray(like))
        return coefficient(like)
    monkeypatch.setattr(S, "_psih_stable_reciprocal", spy)
    actual = np.asarray(S._psih_stable_full(jnp.asarray(z)))
    finite(z); finite(ref); finite(actual)
    assert len(seen) == 1
    # REAL libm/XLA log+pow, reference is well conditioned at |z/L|>=10.
    assert np.all(np.abs(actual.astype(float) - ref[1].astype(float)) <= 2 * np.spacing(np.abs(ref[1])))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_reciprocal_off_and_double_unchanged(monkeypatch, dtype):
    with jax.enable_x64(True):
        argument = jnp.asarray(12., dtype)
        monkeypatch.setenv(KEY, "0")
        assert S._psih_stable_reciprocal(argument) == 1.0 / 1.1
        if dtype == jnp.float64:
            monkeypatch.setenv(KEY, "1")
            assert S._psih_stable_reciprocal(argument) == 1.0 / 1.1


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_finite_gate_rejects_poison(value):
    with pytest.raises(AssertionError, match="non-finite"):
        finite(np.asarray([value], np.float32))
