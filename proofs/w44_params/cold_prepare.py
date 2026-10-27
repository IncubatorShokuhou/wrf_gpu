"""Compile/load one OFF or ON PROD domain into the shared cache without stepping.

The operands use the existing domain-spec builder and original operational JIT.
No model function is replaced. A CPU launch smoke uses five real Thompson columns
through the same load/lower/compile/export path; it is launcher evidence only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from phase_journal import PhaseJournal, write_receipt


from family_work import defaults, bind, cpu_operands
from launch_environment import select_device


def prepare(name, function, operands, cheap_key, journal, static_kwargs=None):
    static_kwargs = {} if static_kwargs is None else static_kwargs
    from gpuwrf.runtime import aot_executable as ax, aot_precompile as ap, aot_cheap_key as ck
    with journal.phase("aot_load", name=name, cheap_key=cheap_key) as record:
        call, status = ap.load_domain_blob(name, cheap_key=cheap_key, return_status=True)
        record.update(status)
    if call is not None:
        return call, status
    with journal.phase("lower", name=name) as record:
        lowered = function.lower(*operands, **static_kwargs)
        hlo = ax.hlo_sha256_from_lowered(lowered)
        assert hlo, "missing lower-only HLO digest"
        record["hlo_sha256"] = hlo
    with journal.phase("hlo_aot_load", name=name, hlo_sha256=hlo) as record:
        call, status = ap.load_domain_blob(name, hlo_sha256=hlo, return_status=True)
        record.update(status)
    if call is not None:
        with journal.phase("alias", name=name) as record:
            record.update(ap.alias_cheap_key_to_hlo(name, cheap_key, hlo))
        status.update(cheap_key=cheap_key, hlo_sha256=hlo)
        return call, status
    with journal.phase("compile", name=name):
        call = lowered.compile(ax.export_compile_options())
    with journal.phase("aot_export", name=name) as record:
        status = ap._serialize_domain_blob(name, call, None, hlo_sha256=hlo,
                                          lowered=lowered, cheap_key=cheap_key,
                                          key_schema=ck.KEY_SCHEMA)
        record.update(status)
        assert status["aot_written"], status
    status.update(loaded=False, hlo_sha256=hlo, cheap_key=cheap_key)
    return call, status


def domain_operands(tree, initial, domain):
    """Use the existing domain specs; prepare records as the runtime does."""
    import jax.numpy as jnp
    from gpuwrf.runtime import aot_precompile as ap, aot_cheap_key as ck
    from gpuwrf.runtime import domain_tree as dt
    from gpuwrf.runtime import operational_mode as om
    # Match prewarm_runtime_threads: a child is advanced only after the real
    # force-down replaces boundary placeholders, including their placement.
    # These local carries are discarded; no parent timestep is executed.
    runtime_carries = dict(initial)
    if tree.hierarchy.parent(domain) is not None:
        for parent, edges in tree.edges.items():
            for edge in edges:
                runtime_carries[edge.child] = dt._operational_force(
                    edge, runtime_carries[parent], runtime_carries[edge.child])
    spec = next(spec for spec in ap.build_defused_specs(tree, carries=runtime_carries) if spec.name == domain)
    carry = runtime_carries[domain]
    clock = om.build_clock_base(spec.namelist)
    records = getattr(spec.namelist, "record_windows", None)
    if records is not None:
        carry, clock = records.prepare(carry, clock, 1, spec.n_steps)
    function = om._advance_chunk_fori
    operands = (carry, spec.namelist, jnp.asarray(1, jnp.int32), clock)
    kwargs = dict(n_steps=spec.n_steps, cadence=spec.cadence)
    key = ck.cheap_key(function, operands, kwargs, spec.namelist)
    assert key, "missing operational domain cheap key"
    return function, operands, key, kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("off", "on"), required=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--family", choices=("diffusion", "mynn"), required=True)
    parser.add_argument("--domain", choices=("d01", "d02"), required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    args.out.mkdir(parents=True, exist_ok=False)
    journal = PhaseJournal(args.out / "phases.jsonl")
    receipt = dict(pin=pin, variant=args.variant, domain=args.domain, family=args.family, source=str(source), status="STARTED",
                   gpu_executed=False, cpu_smoke=args.cpu_smoke,
                   cache=os.environ.get("JAX_COMPILATION_CACHE_DIR"),
                   environment={key: os.environ.get(key) for key in sorted(os.environ)
                                if key.startswith(("GPUWRF_", "JAX_", "XLA_", "TRITON_", "CUDA_"))})
    write_receipt(args.out / "receipt.json", receipt)
    receipt.update(select_device(args.cpu_smoke))
    sys.path.insert(0, str(source / "src"))
    try:
        with journal.phase("imports_and_device"):
            import jax
            from gpuwrf._fast_defaults import apply_fast_path_defaults
            apply_fast_path_defaults()
            assert jax.devices()[0].platform == ("cpu" if args.cpu_smoke else "gpu")
            receipt.update(platform=jax.devices()[0].platform, gpu_executed=not args.cpu_smoke)
        if args.cpu_smoke:
            with journal.phase("model_inputs"):
                function, operands, key, receipt["mini_work"] = cpu_operands(source, args.family, args.variant)
            name = "launch_smoke_" + args.family + "_" + args.variant
            kwargs = {}
        else:
            import dataclasses
            from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
            from gpuwrf.runtime import domain_tree as dt

            cfg = NestedPipelineConfig(args.input_dir, args.out / "unused-output", args.out / "unused-proof", hours=3, max_dom=2)
            with journal.phase("load_domains"):
                hierarchy, bundles, meta, start, steps, initial = _load_domains(cfg, ("d01", "d02"))
            with journal.phase("bind_and_prepare"):
                if args.variant == "on":
                    bundles = {name: dataclasses.replace(bundle, namelist=bind(bundle.namelist, args.family))
                               for name, bundle in bundles.items()}
                nesting = meta["nesting"]
                tree = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=nesting["feedback"],
                    feedback_spec_zone=nesting["spec_zone"], feedback_smooth_option=nesting["smooth_option"])
                from product_schedule import product_schedule
                _, receipt["schedule"] = product_schedule(args.input_dir, ("d01", "d02"), steps, hours=3)
                assert receipt["schedule"]["output_cadence_steps"] == {"d01": 67, "d02": 200}
                assert receipt["schedule"]["event_aware_fusion_k"] == 0
                function, operands, key, kwargs = domain_operands(tree, initial, args.domain)
                receipt["work"] = dict(input_dir=str(args.input_dir), dt_s=steps, start=start.isoformat(), meta=meta)
            name = args.domain
            receipt["compile_work"] = dict(program=name, **kwargs)
        call, status = prepare(name, function, operands, key, journal, kwargs)
        receipt["aot"] = status
        receipt["stepping_executed"] = False
        if args.cpu_smoke:
            import numpy as np
            with journal.phase("mini_model_execute"):
                result = jax.block_until_ready(call(*operands))
                assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result))
                receipt["output_leaves"] = len(jax.tree.leaves(result))
            # A fresh load must execute the same real model with exact results.
            loaded, second = prepare(name, function, operands, key, journal)
            assert second["loaded"], second
            with journal.phase("mini_model_replay"):
                replay = jax.block_until_ready(loaded(*operands))
                assert all(np.asarray(a).tobytes() == np.asarray(b).tobytes()
                           for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(replay), strict=True))
            receipt.update(stepping_executed=True, replay_byte_exact=True,
                           scope="CPU launcher/AOT/real-column model smoke; not GPU/root/fidelity validation")
        receipt["status"] = "PASS"
    except BaseException as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_receipt(args.out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
