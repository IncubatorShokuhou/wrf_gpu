"""One sealed actual c10/c18 segment on GPU; smoke, not WRF fidelity or S1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import faulthandler

sys.dont_write_bytecode = True
source, case, out = map(Path, sys.argv[1:4])
replay_blob = Path(sys.argv[4]) if len(sys.argv) == 5 else None
assert case.name in ("c10", "c18")
assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
assert subprocess.run(["git", "-C", str(source), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 0
assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
out.mkdir(parents=True, exist_ok=False)
stack_log = (out / "progress_stacks.log").open("w")
faulthandler.dump_traceback_later(45, repeat=True, file=stack_log)
started = time.monotonic()
record = dict(case=case.name, scope="actual GPU segment smoke, not WRF fidelity/S1",
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
              source_tree=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD:src"], text=True).strip(),
              cache=str(out / "cache"), native_replay=replay_blob is not None,
              work="n_steps200/radiation33/FAST1CENSUS1SLIM0AOTSLIM0MPRE1/AUTOTUNE0", phase="start")

def save():
    record["wall_s"] = time.monotonic() - started
    (out / "receipt.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(record["phase"], record["wall_s"], flush=True)

save()
try:
    inputs = json.loads((case / "inputs.json").read_text())
    record["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (case / "input").iterdir() if p.is_file()}
    assert record["input_sha256"] == inputs["input_sha256"]
    sys.path.insert(0, str(source / "src"))
    import gpuwrf
    import jax
    import jax.numpy as jnp
    import numpy as np
    assert jax.devices()[0].platform == "gpu"
    assert str(Path(gpuwrf.__file__).resolve()).startswith(str(source / "src"))
    record.update(platform=jax.devices()[0].platform, device=str(jax.devices()[0]), phase="gpu_asserted")
    save()
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.io.gen2_accessor import Gen2Run
    names = tuple(inputs["domains"])
    assert len(names) == 1 and names[0] == "d01"
    cfg = pipeline.NestedPipelineConfig(case / "input", out / "unused_output", out / "unused_proof", hours=inputs["hours"], max_dom=1)
    _, bundles, _, _, dts, carries = pipeline._load_domains(cfg, names)
    nl = bundles["d01"].namelist
    op._resolve_operational_suite(nl)
    cadence, _ = pipeline._output_cadence_steps_by_domain(Gen2Run(case / "input"), names, dts)
    assert cadence["d01"] == 200 and nl.radiation_cadence_steps == 33
    # Match DomainTree's real admission boundary. FAST1 donates the carry;
    # initial RK references share State buffers until this allocation-only prep.
    carry = op.unalias_donated_carry(carries["d01"], by_buffer=True)
    pointers = [v.unsafe_buffer_pointer() for v in jax.tree.leaves(carry)
                if isinstance(v, jax.Array) and v.size]
    if op._CARRY_DONATE:
        assert len(pointers) == len(set(pointers)), "carry must have one owner per buffer"
    record["production_unalias_admission"] = True
    args = (carry, nl, jnp.asarray(1, jnp.int32), op.build_clock_base(nl))
    kw = dict(n_steps=200, cadence=33)
    record["phase"] = "compile"; save()
    tick = time.monotonic()
    lowered = op._advance_chunk_fori.lower(*args, **kw)
    if replay_blob is None:
        compiled = lowered.compile(compiler_options={"xla_gpu_autotune_level": 0})
        record["compile_s"] = time.monotonic() - tick
        executable = compiled.runtime_executable()
        blob = executable.serialize()
    else:
        # The saved real-GPU native executable already paid its cold compile.
        # Recover only its unchanged call tree from this identical-source lower;
        # no root backend compile or new arithmetic is permitted on this path.
        from gpuwrf.runtime import aot_executable as ax
        original = json.loads((out.parent / "gpu-c10/receipt.json").read_text())
        assert case.name == "c10" and original["source_commit"] == record["source_commit"]
        assert original["input_sha256"] == record["input_sha256"]
        blob = replay_blob.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == original["native_sha256"]
        assert not lowered._lowering.const_args, "no uncaptured hoisted constants"
        meta = ax.AotMeta(
            in_tree=lowered.in_tree, out_tree=lowered.out_tree,
            kept_var_idx=tuple(sorted(lowered._lowering.compile_args["kept_var_idx"])),
            in_avals=ax._compiled_in_aval_records(lowered),
            hlo_sha256=ax.hlo_sha256_from_lowered(lowered),
            fingerprint=ax.target_fingerprint(jax.devices()[0]),
            blob_sha256=hashlib.sha256(blob).hexdigest())
        compiled = ax.load(blob, meta, jax.devices()[0])
        executable = compiled.loaded_executable
        record.update(compile_s=0.0, lower_and_native_load_s=time.monotonic() - tick,
                      native_replayed_from=str(replay_blob), root_backend_compile=False)
    native = out / "cache/aot/gpu"
    if replay_blob is None:
        native.mkdir(parents=True)
        (native / "d01.xlaexec").write_bytes(blob)
    else:
        record["cache"] = str(replay_blob.parent)
        record["e41_reuse_native_sha256"] = hashlib.sha256(blob).hexdigest()
    record["native_sha256"] = hashlib.sha256(blob).hexdigest()
    stats = executable.get_compiled_memory_stats()
    record["memory"] = {k: getattr(stats, k) for k in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    if replay_blob is not None:
        assert record["memory"] == original["memory"], "loaded native memory contract must match"
    record["phase"] = "execute"; save()
    tick = time.monotonic()
    result = jax.block_until_ready(compiled(*args, **kw))
    record["segment_s"] = time.monotonic() - tick
    # These counters were already resident in the CENSUS1 compiled program.
    # Read them once at the completed segment boundary, outside its timing.
    from gpuwrf.diagnostics import census
    assert result.census is not None and result.census.f2_counts is not None
    host = jax.device_get(result.census)
    guards = {name: dict(zip(census.EVENTS, map(int, values)))
              for name, values in zip(census.GUARDS, host.guards)}
    work = dict(zip(census.WORK, map(int, host.work)))
    assert work["steps"] == 200
    f2 = census.f2_record(host, result.state.qv.shape)
    counter_record = dict(actual_work=work, guards=guards, f2=f2,
                          resolved_namelist=census.resolved_namelist(nl, output_cadence_steps=200, output_set=()))
    (out / "census.json").write_text(json.dumps(counter_record, indent=2, allow_nan=False) + "\n")
    (out / "f2_negative_summary.json").write_text(json.dumps(
        census.f2_negative_summary(host, result.state.qv.shape), indent=2, allow_nan=False) + "\n")
    record["premask_nonfinite"] = sum(row["nonfinite"] for row in guards.values()) + sum(
        row["count"] for row in f2["events"] if row["event"] == "nonfinite")
    record["f2_event_counts"] = {event: sum(row["count"] for row in f2["events"] if row["event"] == event)
                                 for event in census.F2_EVENTS}
    record["actual_work"] = work
    save()
    assert record["premask_nonfinite"] == 0
    leaves = [np.asarray(v) for v in jax.tree.leaves(result) if hasattr(v, "dtype")]
    record["nonfinite"] = sum(int(np.count_nonzero(~np.isfinite(v))) for v in leaves)
    assert record["nonfinite"] == 0
    record["state_leaves"] = len(leaves)
    record["phase"] = "PASS"; save()
except BaseException as exc:
    record["error"] = repr(exc)
    record["phase"] = "FAIL"; save()
    raise
