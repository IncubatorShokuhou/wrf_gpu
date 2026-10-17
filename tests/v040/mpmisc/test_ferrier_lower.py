"""CPU-client CUDA lowering of both operational Ferrier kernels (E115).

These tests exercise actual Triton, not the CPU Pallas interpreter; numerical
qualification remains the separate original-WRF column gate.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.ferrier.native import native_column


@pytest.mark.parametrize("option", [5, 15])
def test_ferrier_native_cuda_lower(option, monkeypatch):
    from gpuwrf.backends import pallas
    monkeypatch.setattr(pallas, "cuda_native", lambda: True)
    shape = (44, 1, 33)  # partial second tile exercises the native masks
    fields = [jax.ShapeDtypeStruct(shape, np.float32)] * 9
    step = jax.ShapeDtypeStruct((), np.float32)
    with jax.enable_x64(True):
        traced = jax.jit(partial(native_column, option=option, interpret=False)).trace(*fields, step)
        ir = traced.lower(lowering_platforms=("cuda",)).as_text()
    assert "__gpu$xla.gpu.triton" in ir
    assert f"ferrier_mp{option}_columns" in ir
    # WRF CONDENSE/DEPOSIT HIGH_PRES locals remain DOUBLE, including inside
    # the level loop; field inputs/outputs remain REAL32.
    # StableHLO serializes the Triton body as bytecode. Inspect the same
    # Pallas jaxpr as textual TTIR, rather than searching opaque bytes.
    from jax._src.pallas.triton import lowering
    calls = [e for e in traced.jaxpr.eqns if e.primitive.name == "pallas_call"]
    assert len(calls) == 1
    params = calls[0].params
    ttir = str(lowering.lower_jaxpr_to_triton_module(params["jaxpr"], params["grid_mapping"], "cuda", 120).module)
    assert "f64" in ttir
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(traced.out_info))


@pytest.mark.parametrize("option", [85, 95])
def test_unadmitted_ferrier_family_stays_refused(option):
    from gpuwrf.io.scheme_catalog import classify_scheme, SupportStatus
    assert classify_scheme("mp_physics", option).status != SupportStatus.IMPLEMENTED
