"""GPUWRF_STEP_CMDBUF: XLA command-buffer options for the stepping programs only (o5-perf H-CB).

The switch is resolved at import, so every case runs in a fresh CPU subprocess. The options are applied ONLY at
the AOT compile sites of the stepping programs (``operational_mode.step_compile_options``); every jit object
stays the switch-off object, because JAX refuses compiler_options on a jit that is traced inside another jit
(fused cascades, vmapped lanes, tile drivers, eval_shape/make_jaxpr wrappers; four-eyes F1/F5). The compiled
check serializes a CPU executable compiled with ``step_compile_options`` and reads the env-option overrides
stored in its CompileOptions: the option must be present exactly when the switch is on.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json, sys
import jax, jax.numpy as jnp
from gpuwrf.runtime import operational_mode as op, aot_executable as ax
out = {"mode": op._STEP_CMDBUF, "options": op.STEP_COMPILER_OPTIONS, "merged": op.step_compile_options({"k": 1})}
compiled = jax.jit(lambda x: jnp.sin(x) + 1.0).trace(jnp.ones(8, jnp.float32)).lower().compile(
    op.step_compile_options(None))
opts = ax._cpu_compile_options(compiled.runtime_executable().serialize())
out["overrides"] = {k: str(v) for k, v in opts.env_option_overrides}
info = lambda f: {"kvs": [list(kv) for kv in f._jit_info.compiler_options_kvs],
                  "donate": list(f._jit_info.donate_argnums)}
out["fori"] = info(op._advance_chunk_fori)
out["scan"] = info(op._advance_chunk_static_scan)
print("RESULT " + json.dumps(out))
"""


def _run(mode: str | None, **extra: str) -> subprocess.CompletedProcess:
    # tests/conftest.py exports GPUWRF_FAST_DEFAULTS=0 for CPU pytest; every case states the master switch itself.
    env = {k: v for k, v in os.environ.items() if k not in {"GPUWRF_STEP_CMDBUF", "GPUWRF_FAST_DEFAULTS",
                                                            "GPUWRF_CARRY_DONATE"}}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"), **extra)
    if mode is not None:
        env["GPUWRF_STEP_CMDBUF"] = mode
    return subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=600)


def _result(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def test_default_off_adds_no_options():
    res = _result(_run(None, GPUWRF_FAST_DEFAULTS="1"))
    assert res["mode"] == "off" and res["options"] is None and res["merged"] == {"k": 1}
    assert "xla_gpu_graph_min_graph_size" not in res["overrides"]


def test_master_opt_out_adds_no_options():
    res = _result(_run(None, GPUWRF_FAST_DEFAULTS="0"))
    assert res["mode"] == "off" and res["options"] is None
    assert "xla_gpu_graph_min_graph_size" not in res["overrides"]
    assert "xla_gpu_enable_command_buffer" not in res["overrides"]


def test_v1_sets_min_graph_size_in_compiled_options():
    res = _result(_run("v1", GPUWRF_FAST_DEFAULTS="0"))
    assert res["options"] == {"xla_gpu_graph_min_graph_size": 1}
    assert res["merged"] == {"k": 1, "xla_gpu_graph_min_graph_size": 1}
    assert res["overrides"].get("xla_gpu_graph_min_graph_size") == "1"


def test_v2_drops_while_from_capture_set():
    res = _result(_run("v2", GPUWRF_FAST_DEFAULTS="0"))
    types = res["overrides"].get("xla_gpu_enable_command_buffer", "")
    assert types and "WHILE" not in types.split(",") and "CONDITIONAL" in types.split(",")


def test_invalid_value_refuses_loudly():
    proc = _run("v3", GPUWRF_FAST_DEFAULTS="0")
    assert proc.returncode != 0 and "GPUWRF_STEP_CMDBUF" in proc.stderr


@pytest.mark.parametrize("mode", [None, "v1", "v2"])
def test_step_jit_objects_never_carry_compiler_options(mode):
    """F1/F5: a jit with compiler_options cannot be traced inside another jit, so no mode may put them on the jits;
    donation (shipping set) is unchanged."""
    for extra, donate in (({}, []), ({"GPUWRF_CARRY_DONATE": "1"}, [0])):
        res = _result(_run(mode, GPUWRF_FAST_DEFAULTS="0", **extra))
        assert res["fori"]["kvs"] == [] and res["scan"]["kvs"] == [], (mode, extra)
        assert res["fori"]["donate"] == donate, (mode, extra)


def test_nested_option_jit_is_refused_by_jax():
    """Precondition of the design: JAX refuses a nested jit that carries compiler_options."""
    import jax
    import jax.numpy as jnp

    inner = jax.jit(lambda x: x + 1.0, compiler_options={"xla_gpu_graph_min_graph_size": 1})
    with pytest.raises(Exception, match="compiler_options"):
        jax.jit(lambda x: inner(x) * 2.0)(jnp.ones(4))


# Every AOT compile of a stepping program must pass through step_compile_options; a new compile site has to make
# that decision explicitly (this table), so the test fails when a site is added or a wrapper is dropped.
COMPILE_SITES = {
    "src/gpuwrf/runtime/domain_tree.py": 2,      # per-domain advance + fused cascade
    "src/gpuwrf/runtime/aot_precompile.py": 1,   # cold threaded _advance_chunk_fori jobs
}


def _compile_calls(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "lowered"):
            yield node


def _called_name(node) -> str | None:
    func = getattr(node, "func", None)
    return getattr(func, "attr", None) or getattr(func, "id", None)


@pytest.mark.parametrize("rel", sorted(COMPILE_SITES))
def test_stepping_aot_compile_sites_apply_step_options(rel):
    calls = list(_compile_calls(ROOT / rel))
    wrapped = [c for c in calls if c.args and _called_name(c.args[0]) == "step_compile_options"]
    assert len(wrapped) == COMPILE_SITES[rel], (rel, len(wrapped), len(calls))
    if rel.endswith("domain_tree.py"):
        assert len(calls) == len(wrapped), "an unwrapped lowered.compile appeared in domain_tree"


def test_precompile_primitive_receives_step_options_for_stepping_programs():
    tree = ast.parse((ROOT / "src/gpuwrf/runtime/aot_precompile.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _called_name(n) == "precompile"
             and n.args and getattr(n.args[0], "id", None) == "_advance_chunk_fori"]
    assert len(calls) == 2
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "compile_options" in kw and _called_name(kw["compile_options"]) == "step_compile_options"


FUSED = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import jax.numpy as jnp, numpy as np
import test_fused_cascade_case_blind as fc
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.nesting.boundary_construction import build_nest_force_weights
from gpuwrf.runtime import domain_tree as dt, aot_executable as ax, operational_mode as op
from gpuwrf.runtime.aot_key_probe import capture_call
from gpuwrf.runtime.operational_mode import _initial_carry_for_run
date, value = fc.CASES[0]
grid = GridSpec.canary_3km_template(); shapes = _state_field_shapes(grid)
state = State(**{f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f)) for f, s in shapes.items()})
parent = fc._namelist(grid, shapes, date=date, case_value=value, dt_s=fc.PARENT_DT)
child = fc._namelist(grid, shapes, date=date, case_value=2.0 * value, dt_s=fc.CHILD_DT)
weights = build_nest_force_weights(parent_grid_ratio=fc.RATIO, i_parent_start=3, j_parent_start=3,
                                   parent_grid=grid, child_grid=grid)
program = dt._build_fused_cascade_program(
    parent_name="d02", parent_namelist=parent, parent_cadence=4, child_names=("d03",), child_namelists=(child,),
    child_weights=(weights,), child_bdy_widths=(int(state.u_bdy.shape[2]),), child_ratios=(fc.RATIO,),
    child_cadences=(4 * fc.RATIO,))
carries = (_initial_carry_for_run(state, parent), _initial_carry_for_run(state, child))
record = capture_call(program, carries[0], (carries[1],), 1, (1,), lower=True)
assert record is not None and record["program"] == "fused"
out = {"hlo": hashlib.sha256(record["lowered"].as_text().encode()).hexdigest()}
if len(sys.argv) > 2:
    compiled = record["lowered"].compile(op.step_compile_options(ax.export_compile_options()))
    opts = ax._cpu_compile_options(compiled.runtime_executable().serialize())
    out["overrides"] = {k: str(v) for k, v in opts.env_option_overrides}
print("RESULT " + json.dumps(out))
"""


def _fused(mode: str | None, compile_: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in {"GPUWRF_STEP_CMDBUF", "GPUWRF_FAST_DEFAULTS",
                                                            "GPUWRF_CARRY_DONATE"}}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"), GPUWRF_FAST_DEFAULTS="0",
               GPUWRF_CARRY_DONATE="1", GPUWRF_NESTED_AOT="1")
    env.pop("GPUWRF_AOT_VERIFY", None)
    if mode is not None:
        env["GPUWRF_STEP_CMDBUF"] = mode
    argv = [sys.executable, "-c", FUSED, str(ROOT / "tests")] + (["compile"] if compile_ else [])
    return _result(subprocess.run(argv, env=env, capture_output=True, text=True, timeout=1800))


@pytest.mark.heavy  # ~6.9 GB RSS (CPU compile of the fused cascade): run with --run-heavy in a >= 12G RAM slot
def test_fused_cascade_compiles_under_v1_with_unchanged_hlo():
    """The WN3-type fused cascade (child advances traced inside fused_jit) under the shipping set
    (v1 + CARRY_DONATE): lowers, its lowered HLO equals OFF, and the fused executable compiled with the
    compile-site options carries min graph size 1."""
    off = _fused(None, compile_=False)
    v1 = _fused("v1", compile_=True)
    assert v1["hlo"] == off["hlo"]
    assert v1["overrides"].get("xla_gpu_graph_min_graph_size") == "1"

# --- F6 (four-eyes 01:07Z): OFF and v1 stepping executables share the lowered HLO; the HLO address must differ ----
AOT_PROBE = r"""
import json, sys
from pathlib import Path
from gpuwrf.runtime import aot_precompile as ap
root, action = Path(sys.argv[1]), sys.argv[2]
rel = lambda p: str(p[0].relative_to(root)) if p else None
H = "ab" * 32
if action == "address":
    print("RESULT " + json.dumps({
        "d02_hlo": rel(ap._aot_blob_paths("d02", root, hlo_sha256=H)),
        "fused_hlo": rel(ap._aot_blob_paths("fused_d02", root, hlo_sha256=H)),
        "init_hlo": rel(ap._aot_blob_paths("init_solve_rrtmg_sw_column", root, hlo_sha256=H)),
        "d02_key": rel(ap._aot_blob_paths("d02", root, cheap_key="kk"))}))
elif action == "place":
    blob, meta = ap._aot_blob_paths("d02", root, hlo_sha256=H)
    blob.parent.mkdir(parents=True, exist_ok=True); blob.write_bytes(b"blob"); meta.write_bytes(b"meta")
    print("RESULT " + json.dumps({"placed": rel((blob, meta))}))
else:
    call, status = ap.load_domain_blob("d02", root, hlo_sha256=H, return_status=True)
    print("RESULT " + json.dumps({"call": call is not None, "source": status.get("source")}))
"""


def _aot(mode, root, action):
    env = {k: v for k, v in os.environ.items() if k not in {"GPUWRF_STEP_CMDBUF", "GPUWRF_FAST_DEFAULTS"}}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"), GPUWRF_FAST_DEFAULTS="0")
    if mode:
        env["GPUWRF_STEP_CMDBUF"] = mode
    return _result(subprocess.run([sys.executable, "-c", AOT_PROBE, str(root), action], env=env, capture_output=True,
                                  text=True, timeout=600))


def test_stepping_hlo_address_is_variant_aware(tmp_path):
    off, v1 = _aot(None, tmp_path, "address"), _aot("v1", tmp_path, "address")
    assert off["d02_hlo"].endswith("/d02/" + "ab" * 32 + ".xlaexec")
    assert v1["d02_hlo"].endswith("/d02/" + "ab" * 32 + ".cmdbuf-v1.xlaexec")
    assert v1["fused_hlo"].endswith("/fused_d02/" + "ab" * 32 + ".cmdbuf-v1.xlaexec")
    assert off["init_hlo"] == v1["init_hlo"]      # non-stepping programs keep sharing
    assert off["d02_key"] == v1["d02_key"]        # cheap keys already differ through the switch constant


def test_off_blob_is_never_loaded_as_v1_by_hlo(tmp_path):
    """Serialize-OFF / load-as-V1-by-HLO must miss (four-eyes F6); the OFF process still finds its own blob."""
    _aot(None, tmp_path, "place")
    off, v1 = _aot(None, tmp_path, "load"), _aot("v1", tmp_path, "load")
    assert off["source"] != "fallback:missing"    # the placed OFF artifact is found (dummy bytes: load error)
    assert v1["call"] is False and v1["source"] == "fallback:missing"
