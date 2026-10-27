"""scripts/off_identity_probe.py + off_identity_compare.py (critic cr-o5-port-probe F1-F4).

Light tests (seconds, CPU): the E161 mechanism on a toy program (old canon-first order BLIND, native-first and
clear_caches COMPLETE), the constant-value blindness of ``as_hlo_text()`` and its full-constant fix, the name
self-check, comparator refusals, and an ERROR receipt when case loading fails. Heavy tests (opt in with
--run-heavy; ~3-4 min and ~3 GB per probe process) run the real PROD case in both orders and nested.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("off_identity_probe")
compare = _load("off_identity_compare")
PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


def _kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * jnp.float32(3)


@jax.jit
def _inner(x):  # backend-dependent dispatch decided at trace time, like the product kernels
    interpret = jax.default_backend() == "cpu"
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), interpret=interpret,
                          compiler_params=pt.CompilerParams(num_warps=4), name="probe_toy_kernel")(x)


def _outer(x):
    return _inner(x) + jnp.float32(1)


X = jax.ShapeDtypeStruct((256,), jnp.float32)


def _native(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with probe.KernelCensus("toy") as census:
        traced = jax.jit(_outer).trace(X)
        stats = probe.pallas_stats(traced.jaxpr)
        hlo = traced.lower(lowering_platforms=("cuda",)).compiler_ir("hlo")
    return stats, census.kernels, probe.hlo_hashes(hlo)["triton_custom_calls"]


@pytest.fixture(autouse=True)
def _fresh():
    jax.clear_caches()
    yield
    jax.clear_caches()


def test_old_canon_first_order_is_blind(monkeypatch):
    jax.make_jaxpr(_outer)(X)  # canonical pass on the CPU default backend (old probe order)
    stats, kernels, triton = _native(monkeypatch)
    assert stats["pallas_calls"] == 1 and stats["interpreted"] == 1, stats
    assert stats["interpreted_names"] == {"probe_toy_kernel": 1} and stats["native_names"] == {}
    assert triton == 0 and kernels == []  # the CUDA program silently lacks the kernel


def test_native_first_order_sees_the_kernel(monkeypatch):
    stats, kernels, triton = _native(monkeypatch)
    assert stats["interpreted"] == 0 and stats["native_names"] == {"probe_toy_kernel": 1}
    assert triton == 1 and len(kernels) == 1 and kernels[0]["calls"] == 1
    # A3: the census reads num_warps/num_stages from JAX's lowering frame; a rename must fail here.
    assert kernels[0]["num_warps"] == 4 and kernels[0]["num_stages"] == 3 and kernels[0]["func"] == "probe_toy_kernel"
    assert probe.name_selfcheck(stats, kernels) == []


def test_clear_caches_between_passes_restores_the_kernel(monkeypatch):
    jax.make_jaxpr(_outer)(X)
    jax.clear_caches()
    stats, _, triton = _native(monkeypatch)
    assert stats["interpreted"] == 0 and triton == 1


def test_native_pass_clears_caches_first():
    """D1: the native pass must start with jax.clear_caches() (load-phase CPU traces, critic F2)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(probe.main))
    native = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "native")
    calls = [ast.unparse(n.func) for n in ast.walk(native) if isinstance(n, ast.Call)]
    assert "jax.clear_caches" in calls, calls
    first = native.body[0]
    assert isinstance(first, ast.If) and ast.unparse(first.body[0].value.func) == "jax.clear_caches"


def test_name_selfcheck_flags_both_directions():
    stats = {"native_names": {"a": 1, "b": 2}}
    assert probe.name_selfcheck(stats, [{"func": "a"}, {"func": "b"}]) == []
    missing = probe.name_selfcheck(stats, [{"func": "a"}])
    assert len(missing) == 1 and "without a Triton lowering" in missing[0]  # D5: explicit assertion form
    extra = probe.name_selfcheck({"native_names": {"a": 1}}, [{"func": "a"}, {"func": "c"}])
    assert len(extra) == 1 and "without a native pallas_call" in extra[0]


def _table_program(table):
    def f(x):
        return x * jnp.asarray(table) + jnp.float32(1)
    spec = jax.ShapeDtypeStruct(table.shape, jnp.float32)
    return jax.jit(f).trace(spec).lower().compiler_ir("hlo"), jax.make_jaxpr(f)(jnp.zeros(table.shape, jnp.float32))


def test_constant_value_mutant_is_seen_only_by_the_full_hashes():
    table = np.linspace(0, 1, 585 * 16, dtype=np.float32).reshape(585, 16)  # RRTMG-table-sized constant
    mutant = table.copy()
    mutant[17, 3] = np.float32(0.75)
    (h_a, j_a), (h_b, j_b) = _table_program(table), _table_program(mutant)
    a, b = probe.hlo_hashes(h_a), probe.hlo_hashes(h_b)
    assert a["elided_constants"] >= 1
    assert a["raw_hlo_sha256"] == b["raw_hlo_sha256"]  # the legacy identity is blind to the value (F1)
    assert a["raw_hlo_full_sha256"] != b["raw_hlo_full_sha256"]
    assert probe.consts_sha256(j_a) != probe.consts_sha256(j_b)
    assert probe.canon_text(j_a) == probe.canon_text(j_b)  # the canonical jaxpr text is blind as well


def _receipt(**over):
    rec = {"schema": 2, "verdict": "COMPLETE", "case": "/case", "covered_programs": ["root_d01_step"], "max_dom": 1,
           "release_defaults_active": True, "operator_env": {},
           "provenance": {"versions": {"jax": "0.10.0"}, "program_env": {"JAX_PLATFORMS": "cpu"}, "realpath": "/t"},
           "canon_sha256": "c", "canon_consts_sha256": "k",
           "programs": {"root_d01_step": {"raw_hlo_full_sha256": "F", "raw_hlo_sha256": "R", "triton_custom_calls": 1,
                                          "n_kernel_bodies": 1, "n_pallas_lowerings": 1,
                                          "lowerings_by_name": {"k": 1}}},
           "kernels": [{"program": "root_d01_step", "ttir_sha16": "t", "ttir_loc_sha16": "l", "num_warps": 4,
                        "num_stages": 3, "calls": 1}]}
    rec.update(over)
    return rec


def test_compare_semantics():
    good = _receipt()
    assert compare.compare(good, copy.deepcopy(good))["verdict"] == "IDENTICAL"
    full = copy.deepcopy(good)
    full["programs"]["root_d01_step"]["raw_hlo_full_sha256"] = "G"  # only a constant value changed
    assert compare.compare(good, full)["verdict"] == "DIFFERENT"
    assert compare.compare(good, _receipt(canon_consts_sha256="x"))["verdict"] == "DIFFERENT"
    loc = _receipt(kernels=[dict(good["kernels"][0], ttir_loc_sha16="m")])
    res = compare.compare(good, loc)
    assert res["verdict"] == "DIFFERENT" and "line shift" in res["hint"]
    for bad in (_receipt(verdict="BLIND"), _receipt(verdict="ERROR"), _receipt(schema=1), _receipt(case="/other"),
                _receipt(covered_programs=["root_d01_step", "child_d02_step"]),
                _receipt(provenance={**good["provenance"], "versions": {"jax": "0.11.0"}}),
                _receipt(operator_env={"GPUWRF_X": "1"}), _receipt(release_defaults_active=False),
                _receipt(order="canon-first"), _receipt(canon_sha256=None, canon_traced=False)):
        assert compare.compare(good, bad)["verdict"] == "INVALID_INPUT", bad


def _run_probe(tmp_path, out, *args, timeout=1500):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GPUWRF_", "JAX_", "XLA_")) and k != "PYTHONPATH"}
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", GPUWRF_JAX_CACHE="0", OFF_PROBE_TMP=str(tmp_path))
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/off_identity_probe.py"), "--alias", str(ROOT),
                           "--out", str(out), *args], env=env, timeout=timeout, capture_output=True, text=True)
    return proc.returncode, json.loads(out.read_text())


def test_failed_case_load_writes_error_receipt(tmp_path):
    empty = tmp_path / "empty_case"
    empty.mkdir()
    out = tmp_path / "err.json"
    code, rec = _run_probe(tmp_path, out, "--case", str(empty), "--no-canon", timeout=600)
    assert out.exists(), "a failed load must still write a receipt (F4)"
    assert code == 1 and rec["verdict"] == "ERROR" and rec["error"]
    assert rec["covered_programs"] == ["root_d01_step"] and rec["programs"] == {}


@pytest.mark.heavy
@pytest.mark.skipif(not PROD.exists(), reason="real PROD case not present")
def test_real_prod_old_order_blind_new_order_complete(tmp_path):
    receipts = {}
    for order in ("canon-first", "native-first"):
        code, receipts[order] = _run_probe(tmp_path, tmp_path / f"{order}.json", "--case", str(PROD),
                                           "--order", order)
    old, new = receipts["canon-first"], receipts["native-first"]
    assert old["release_defaults_active"] and new["release_defaults_active"]
    assert old["verdict"] == "BLIND" and new["verdict"] == "COMPLETE", (old.get("verdict"), new.get("verdict"))
    assert new["triton_custom_calls"] > old["triton_custom_calls"]
    assert new["canon_sha256"] == old["canon_sha256"] and new["canon_consts_sha256"] == old["canon_consts_sha256"]
    assert new["programs"]["root_d01_step"]["elided_constants"] > 100  # the legacy hash elides the tables


CHILD_ONLY = {"b_diff_advect_u_fp32", "b_diff_advect_v_fp32", "b_diff_large_step_u_fp32", "b_diff_large_step_v_fp32",
              "boundary_spec_final_real"}  # in the PROD d02 GPU program, absent from d01 (critic F4)


@pytest.mark.heavy
@pytest.mark.skipif(not PROD.exists(), reason="real PROD case not present")
def test_real_prod_nested_covers_child_only_kernels(tmp_path):
    code, rec = _run_probe(tmp_path, tmp_path / "nested.json", "--case", str(PROD), "--nested", "--no-canon")
    assert code == 0 and rec["verdict"] == "COMPLETE", rec.get("blind_reasons") or rec.get("error")
    assert set(rec["programs"]) == {"root_d01_step", "forcedown_d01_d02", "child_d02_step"}
    child = {k["func"] for k in rec["kernels"] if k["program"] == "child_d02_step"}
    root = {k["func"] for k in rec["kernels"] if k["program"] == "root_d01_step"}
    assert CHILD_ONLY <= child and not (CHILD_ONLY & root), sorted(CHILD_ONLY - child)
