"""Check cold-preparation keys against real PROD runtime call metadata on CPU.

Load real seeded inputs and execute the existing force-down producer. Replace
only the AOT-loaded advance callable with a recorder returning its input carry;
forbid full-advance lowering/fallback. Init/force-down can use CPU JIT. This is
an operand/key test, not model-step execution or physics fidelity evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

from cold_prepare import DEFAULTS, domain_operands
from phase_journal import write_receipt
from product_schedule import product_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1"
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    sys.path.insert(0, str(source / "src"))
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="STARTED", source=str(source), pin=pin,
                   scope=__doc__, gpu_executed=False, full_advance_executed=False,
                   environment={k: os.environ.get(k) for k in (
                       "JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_JAX_CACHE",
                       "GPUWRF_CARRY_DONATE", "GPUWRF_ADVANCE_CHUNK_LOOP")}, variants={})
    write_receipt(args.out / "receipt.json", receipt)
    try:
        import jax
        from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
        from gpuwrf.runtime import aot_executable as ax, aot_cheap_key as ck, aot_precompile as ap
        from gpuwrf.runtime import domain_tree as dt, operational_mode as om
        from gpuwrf.runtime_parameters import bind_domain_parameters
        assert jax.devices()[0].platform == "cpu"
        assert dt._nested_aot_enabled() and not dt._nested_aot_verify_enabled()
        names = ("d01", "d02")
        cfg = NestedPipelineConfig(args.input_dir, args.out / "unused-output",
                                   args.out / "unused-proof", hours=3, max_dom=2)
        hierarchy, bundles, meta, start, steps, initial = _load_domains(cfg, names)
        schedule, receipt["schedule"] = product_schedule(args.input_dir, names, steps, hours=3)
        assert receipt["schedule"]["event_aware_fusion_k"] == 0

        def describe(operands, kwargs, key):
            flat, _ = ax._flatten_call(operands, kwargs)
            records = [ck._leaf_metaty_record(leaf) for leaf in flat]
            leaves = [{name: record.get(name)
                       for name in ("shape", "dtype", "weak_type", "placement_class")}
                      for record in records]
            return dict(cheap_key=key, aval_hash=ck.carry_aval_hash(operands, kwargs),
                        static_hash=ck.static_config_hash(operands[1]),
                        kwargs=dict(kwargs), leaf_count=len(leaves), leaves=leaves)

        def forbidden(*unused, **ignored):
            raise RuntimeError("full advance lowering/execution is forbidden in this metadata probe")

        for variant in ("off", "on"):
            bound = bundles if variant == "off" else {
                name: dataclasses.replace(bundle, namelist=bind_domain_parameters(
                    bundle.namelist, {name: {"thompson_parameters": DEFAULTS}}, name))
                for name, bundle in bundles.items()}
            tree = dt.DomainTree.from_domains(hierarchy, bound)
            expected, actual, calls = {}, {}, []
            original_force = dt._operational_force
            with mock.patch.object(dt, "_operational_force", wraps=original_force) as force_spy:
                for domain in names:
                    function, operands, key, kwargs = domain_operands(tree, initial, domain)
                    assert function is om._advance_chunk_fori
                    expected[domain] = describe(operands, kwargs, key)
                cold_force_calls = force_spy.call_count

                def load_recorder(name, *unused, cheap_key=None, return_status=False, **ignored):
                    assert name in names and cheap_key and return_status
                    def call(*operands, **kwargs):
                        calls.append(name)
                        actual[name] = describe(operands, kwargs, cheap_key)
                        return operands[0]
                    return call, dict(name=name, loaded=True, source="metadata-recorder",
                                      cheap_key=cheap_key, error=None)

                # The real factory and scheduler still prepare record windows,
                # unalias carries, choose programs, and compute their cheap keys.
                # Guard their fail-open paths so a recorder failure cannot step.
                with mock.patch.object(ap, "load_domain_blob", side_effect=load_recorder), \
                     mock.patch.object(om._advance_chunk_fori, "lower", side_effect=forbidden), \
                     mock.patch.object(dt, "_advance_chunk", side_effect=forbidden):
                    result = dt.run_operational_domain_tree(
                        tree, root_steps=1, carries=initial,
                        initial_own_steps={name: 0 for name in names}, **schedule)
                    jax.block_until_ready(result.carries)
                runtime_force_calls = force_spy.call_count - cold_force_calls

            comparisons = {name: dict(
                exact=expected[name] == actual.get(name),
                expected=expected[name], actual=actual.get(name)) for name in names}
            receipt["variants"][variant] = dict(
                calls=calls, own_steps=result.own_steps, cold_force_calls=cold_force_calls,
                runtime_force_calls=runtime_force_calls, comparisons=comparisons)
            write_receipt(args.out / "receipt.json", receipt)
            assert calls == list(names), calls
            assert cold_force_calls == runtime_force_calls == 1
            assert all(row["exact"] for row in comparisons.values()), (
                "cold/runtime metadata differ; see receipt.json", variant)
            assert result.own_steps == {"d01": 1, "d02": 3}
        receipt["tools_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("prepare_inputs_smoke.py", "cold_prepare.py", "product_schedule.py")}
        receipt["status"] = "PASS"
    except BaseException as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_receipt(args.out / "receipt.json", receipt)
    print(json.dumps({"status": "PASS", "variants": list(receipt["variants"]),
                      "full_advance_executed": False, "gpu_executed": False}))


if __name__ == "__main__":
    main()
