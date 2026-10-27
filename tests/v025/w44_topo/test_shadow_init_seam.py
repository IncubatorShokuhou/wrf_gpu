"""Nested init keeps WRF's allocator-zero HGT_SHAD (critic F1).

Both production init sites replace ``radiation_diagnostics`` with the lead-0
Noah-MP init radiation call. That call is not a WRF toposhad history call, so
the seeded zero plane must survive it. The statements are executed from the
real source (AST), with only the costly radiation call stubbed. Deleting the
re-seed makes the held height equal the stub's computed height.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.coupling.topographic_shadow import preserve_initial_height, seed

SRC = Path(__file__).resolve().parents[3] / "src" / "gpuwrf"
SITES = {
    "nested_pipeline": (SRC / "integration" / "nested_pipeline.py", "_load_domains", "carry",
                        dict(namelist="NL", noahmp_land="LAND", _init_kernel_call=None)),
    "initial_feedback": (SRC / "nesting" / "initial_feedback.py", "initialize_two_way_domains", "new",
                         dict(nl="NL", land="LAND", kernel_call=None)),
}


@dataclass(frozen=True)
class _Carry:
    state: object
    radiation_diagnostics: object
    noahmp_land: object = None
    noahmp_rad: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def _diag(height):
    z = jnp.zeros((4, 4), jnp.float32)
    values = [z] * 15
    values[10] = z.astype(jnp.int32)
    return pc.RRTMGRadiationDiagnostics(*values, shadow_height_m=height)


def _init_block(path, function, target):
    """Statements from the init radiation call through the carry replace."""
    tree = ast.parse(path.read_text())
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        body = node.body
        start = next((i for i, s in enumerate(body) if isinstance(s, ast.Assign)
                      and isinstance(s.value, ast.Call) and getattr(s.value.func, "id", "") == "noahmp_initial_rad"), None)
        if start is None:
            continue
        for end in range(start, len(body)):
            s = body[end]
            if (isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == target
                    and isinstance(s.value, ast.Call)
                    and any(k.arg == "radiation_diagnostics" for k in s.value.keywords)):
                module = ast.Module(body=body[start:end + 1], type_ignores=[])
                return compile(module, f"{path}:{function}", "exec")
    raise AssertionError(f"{function}: init radiation replace block not found")


def _run_site(name, computed_height, *, on=True):
    path, function, target, extra = SITES[name]
    calls = []
    computed = _diag(computed_height)

    def noahmp_initial_rad(state, namelist, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("_with_diagnostics") is True
        return "SURFACE", computed

    namelist = SimpleNamespace(topo_shading=int(on), slope_rad=1,
                               boundary_config=SimpleNamespace(force_geopotential=False))
    carry = seed(_Carry(SimpleNamespace(t_skin=jnp.zeros((4, 4), jnp.float64)), _diag(None)), namelist)
    plane = carry.radiation_diagnostics.shadow_height_m
    scope = dict(extra, noahmp_initial_rad=noahmp_initial_rad, **{target: carry})
    exec(_init_block(path, function, target), scope)
    assert len(calls) == 1, "init radiation block did not run"
    return scope[target], computed, plane


@pytest.mark.parametrize("site", sorted(SITES))
def test_nested_init_keeps_seeded_zero_hgt_shad(site):
    result, computed, plane = _run_site(site, jnp.full((4, 4), 2.07, jnp.float32))
    held = result.radiation_diagnostics.shadow_height_m
    np.testing.assert_array_equal(np.asarray(held), np.zeros((4, 4), np.float32))
    assert held is plane, "re-seed must reuse the seeded plane, not allocate"
    assert held.dtype == jnp.float32
    assert result.noahmp_rad == "SURFACE"
    # every other init diagnostic is the init call's value
    for index in range(15):
        assert result.radiation_diagnostics[index] is computed[index]


@pytest.mark.parametrize("site", sorted(SITES))
def test_off_path_keeps_init_diagnostics_object(site):
    result, computed, plane = _run_site(site, None, on=False)
    assert plane is None
    assert result.radiation_diagnostics is computed
