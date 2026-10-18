"""`gpuwrf aot-key` plumbing (CPU, no case load): parser, namelist view, case comparison, no dispatch."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.cli import build_parser
from gpuwrf.contracts.grid import DomainHierarchy, GridSpec
from gpuwrf.runtime import aot_key_cli
from gpuwrf.runtime import domain_tree as dt
from gpuwrf.runtime.aot_key_probe import case_program_keys
from gpuwrf.runtime.operational_mode import OperationalNamelist


def test_aot_key_is_a_gpuwrf_command(tmp_path):
    args = build_parser().parse_args(["aot-key", "--input-dir", str(tmp_path), "--compare-input-dir", str(tmp_path)])
    assert args.func is aot_key_cli.command
    assert args.hours == 3.0 and args.namelist is None


def test_namelist_override_is_a_symlink_view(tmp_path):
    case, other = tmp_path / "case", tmp_path / "other.input"
    case.mkdir()
    (case / "namelist.input").write_text("&domains\n max_dom = 3\n/\n")
    (case / "wrfinput_d01").write_text("x")
    other.write_text("&domains\n max_dom = 2\n/\n")
    assert aot_key_cli._case_dir(case, None, tmp_path) == case
    view = aot_key_cli._case_dir(case, other, tmp_path)
    assert (view / "namelist.input").resolve() == other.resolve()
    assert (view / "wrfinput_d01").resolve() == (case / "wrfinput_d01").resolve()
    assert (case / "namelist.input").read_text().startswith("&domains\n max_dom = 3")


def _record(key, **components):
    base = {name: "same" for name in aot_key_cli._COMPONENTS}
    return {"cheap_key": key, "components": base | components}


def test_comparison_names_the_differing_component():
    a = {"d01": _record("k1"), "fused/d02": _record("k2", fn_identity="case-a")}
    b = {"d01": _record("k1"), "fused/d02": _record("k3", fn_identity="case-b"), "fused/d03": _record("k4")}
    rows = aot_key_cli._compare(a, b)
    assert rows["d01"] == {"shared": True, "differing_components": []}
    assert rows["fused/d02"] == {"shared": False, "differing_components": ["fn_identity"]}
    assert rows["fused/d03"]["shared"] is False


def test_missing_keys_are_never_declared_shared():
    rows = aot_key_cli._compare({"fused/d02": {"cheap_key": None}}, {"fused/d02": {"cheap_key": None}})
    assert rows["fused/d02"]["shared"] is False and rows["fused/d02"]["unavailable"] is True
    rows = aot_key_cli._compare({"d01": _record("k1")}, {"d01": {"cheap_key": None, "unavailable": "x"}})
    assert rows["d01"]["shared"] is False and "compare" in rows["d01"]["reason"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _Carry:
    state: object

    def replace(self, **kw):
        return replace(self, **kw)

    def tree_flatten(self):
        return (self.state,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@pytest.mark.parametrize("env", [{"GPUWRF_NESTED_AOT": "0"}, {"GPUWRF_ADVANCE_CHUNK_LOOP": "scan"}])
def test_probe_never_dispatches_a_model_step(monkeypatch, env):
    """cr-w41-bundlekey P2: AOT off or a non-fori loop mode must not let the dry-run step the model."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.0)
    tree = dt.DomainTree(hierarchy=DomainHierarchy.from_edges(("d01",), ()), domains={"d01": SimpleNamespace(namelist=nl)})
    calls = []
    monkeypatch.setattr(dt, "_resolve_operational_suite", lambda nl: None)

    def model_step(*args, **kwargs):
        calls.append("model-step")
        return args[0]

    monkeypatch.setattr(dt, "_advance_chunk", model_step)
    monkeypatch.setattr(dt, "_operational_fused_cascade_factory", lambda tree: lambda name: None)
    records = case_program_keys(tree, {"d01": _Carry(jnp.ones((1,), jnp.float32))}, components=True)
    assert not calls, "the dry-run stepped the model"
    record = records["d01"]
    if "GPUWRF_ADVANCE_CHUNK_LOOP" in env:  # no AOT for this loop mode: reported, never stepped
        assert record["cheap_key"] is None and record["unavailable"]
    else:  # the probe computes the key an AOT run would use, whatever the caller's env says
        assert record["cheap_key"] is not None
