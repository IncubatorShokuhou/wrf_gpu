"""In-process CPU identity runs for one source tree (one compile per program, shared by every task).

usage: JAX_PLATFORMS=cpu python identity_runner.py SRC_ROOT OUT_DIR TASK [TASK ...]
TASK = pipeline:<case>:<hours>:<max_dom>         -> OUT_DIR/<case>_pipe/wrfout/* + receipt.json
     | steps:<case>:<max_dom>:<first>:<count>     -> OUT_DIR/<case>_s<first>x<count>.npz/.json
     | extend:<case>:<max_dom>:<h1>:<h2>:<ckint> -> OUT_DIR/<case>_ext: run h1 hours with checkpoints every
                                                    ckint root steps, then resume the end checkpoint with
                                                    --extend-run semantics to h2 hours (same stream)
Extra tasks are also read from OUT_DIR/extra_tasks.txt (one per line) after the argv tasks.
pipeline = execute_nested_pipeline (the CLI driver, emit_initial_history=True).
steps    = run_operational_domain_tree for <count> root steps whose clock starts at root step <first>
           (children at the parent-ratio equivalent); the IC is the case's initial state, so a few
           steps cross a boundary record, a window refresh, an aux4 activation or the coverage end.
The resolved gpuwrf package is asserted to live under SRC_ROOT (E159). Each task appends one
line to OUT_DIR/tasks.jsonl (wall, compile count) as soon as it finishes.
"""
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

src, out_dir, tasks = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3:]
sys.path.insert(0, str(src / "src"))
import jax  # noqa: E402
import numpy as np  # noqa: E402
import gpuwrf  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert Path(gpuwrf.__file__).resolve().is_relative_to(src), gpuwrf.__file__
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import domain_tree as dt  # noqa: E402

CASES = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/cases")
out_dir.mkdir(parents=True, exist_ok=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src"], cwd=src, text=True).strip())
compiles = []


class _Grab(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if message.startswith("Compiling") or "Finished XLA compilation" in message:
            compiles.append(message[:300])


handler = _Grab()
for name in ("jax._src.interpreters.pxla", "jax._src.dispatch"):
    logging.getLogger(name).addHandler(handler)
    logging.getLogger(name).setLevel(logging.WARNING)
jax.config.update("jax_log_compiles", True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_pipeline(case, hours, max_dom):
    target = out_dir / f"{case}_pipe"
    target.mkdir(parents=True, exist_ok=False)
    cfg = pipeline.NestedPipelineConfig(CASES / case, target / "wrfout", target / "proof", hours=int(hours),
                                        max_dom=int(max_dom), emit_initial_history=True)
    payload = pipeline.execute_nested_pipeline(cfg)
    files = {p.name: _sha(p) for p in sorted((target / "wrfout").glob("wrfout_*"))}
    receipt = {"verdict": payload.get("verdict"), "files": files,
               "record_windows": {n: (m or {}).get("record_windows")
                                  for n, m in (payload.get("meta", {}).get("domains", {}) or {}).items()}}
    (target / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    return {"files": len(files), "verdict": receipt["verdict"], "record_windows": receipt["record_windows"]}


def run_extend(case, max_dom, h1, h2, ckint):
    from dataclasses import replace
    from gpuwrf.runtime.restart_store import RestartStore

    target = out_dir / f"{case}_ext"
    target.mkdir(parents=True, exist_ok=False)
    cfg = pipeline.NestedPipelineConfig(CASES / case, target / "wrfout", target / "proof", hours=int(h1),
                                        max_dom=int(max_dom), emit_initial_history=True,
                                        checkpoint_dir=target / "ck", checkpoint_interval_steps=int(ckint),
                                        checkpoint_reserve_bytes=0)
    first = pipeline.execute_nested_pipeline(cfg)
    store = RestartStore(cfg.checkpoint_dir, 10**11, 2, 0)
    end = store.latest()
    second = pipeline.execute_nested_pipeline(replace(cfg, hours=int(h2), resume_checkpoint=end, extend_end_time=True))
    files = {p.name: _sha(p) for p in sorted((target / "wrfout").glob("wrfout_*"))}
    receipt = {"first": first.get("verdict"), "second": second.get("verdict"), "resumed_from": str(end),
               "restart": second.get("restart"), "files": files}
    (target / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    return {k: receipt[k] for k in ("first", "second", "resumed_from")} | {"files": len(files)}


def run_steps(case, max_dom, first, count):
    max_dom, first, count = int(max_dom), int(first), int(count)
    names = pipeline.domain_names_for(max_dom)
    cfg = pipeline.NestedPipelineConfig(CASES / case, out_dir / "sc_out", out_dir / "sc_proof", hours=3,
                                        max_dom=max_dom)
    hierarchy, bundles, meta, _run_start, dt_by_domain, carries = pipeline._load_domains(cfg, names)
    tree = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    own = {name: (first - 1) * round(dt_by_domain[names[0]] / dt_by_domain[name]) for name in names}
    result = dt.run_operational_domain_tree(tree, root_steps=count, carries=carries, initial_own_steps=own)
    jax.block_until_ready(jax.tree_util.tree_leaves(result.carries))
    arrays = {}
    for name, carry in result.carries.items():
        for path, leaf in jax.tree_util.tree_flatten_with_path(carry)[0]:
            if isinstance(leaf, jax.Array):
                arrays[f"{name}{jax.tree_util.keystr(path)}"] = np.asarray(leaf)
    stem = out_dir / f"{case}_s{first}x{count}"
    np.savez(stem.with_suffix(".npz"), **arrays)
    floats = [v for v in arrays.values() if np.issubdtype(v.dtype, np.floating)]
    summary = {"own_steps": result.own_steps, "leaves": len(arrays),
               "nonfinite_no_bdy": int(sum(int((~np.isfinite(v)).sum()) for k, v in arrays.items()
                                           if "_bdy" not in k and np.issubdtype(v.dtype, np.floating))),
               "nonfinite_all": int(sum(int((~np.isfinite(v)).sum()) for v in floats)),
               "digest_no_bdy": hashlib.sha256(b"".join(k.encode() + v.tobytes() for k, v in sorted(arrays.items())
                                                        if "_bdy" not in k)).hexdigest(),
               "record_windows": {n: (meta["domains"].get(n) or {}).get("record_windows") for n in names}}
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


extra = out_dir / "extra_tasks.txt"
if extra.exists():
    tasks = tasks + [line.strip() for line in extra.read_text().splitlines() if line.strip()]
RUNNERS = {"pipeline": lambda *a: run_pipeline(*a), "steps": lambda *a: run_steps(*a),
           "extend": lambda *a: run_extend(*a)}
for task in tasks:
    kind, *args = task.split(":")
    compiles.clear()
    t0 = time.time()
    error = None
    try:
        info = RUNNERS[kind](*args)
    except Exception as exc:  # noqa: BLE001 -- record and continue with the next task
        import traceback
        info, error = None, traceback.format_exc()[-3000:]
    line = {"task": task, "src": str(src), "head": head, "dirty": dirty, "wall_s": round(time.time() - t0, 1),
            "compiles": len([c for c in compiles if c.startswith("Compiling")]), "info": info, "error": error}
    with (out_dir / "tasks.jsonl").open("a") as f:
        f.write(json.dumps(line, default=str) + "\n")
    print(json.dumps({k: line[k] for k in ("task", "wall_s", "compiles", "error")}, default=str)[:600], flush=True)
