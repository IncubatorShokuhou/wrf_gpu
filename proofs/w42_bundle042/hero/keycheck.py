"""w42-harness: gate that an arm environment hits the sealed bundle (AOT trace-env hash + C-auto plan key).

usage: python keycheck.py [--emulate-cpu] PLAN_DIR OUT_JSON NAME=ENV_DUMP=ARGV_JSON ...
ENV_DUMP is an `env -0` dump of the environment the measured process starts with (session: measured_env / bundle_env),
ARGV_JSON the `gpuwrf.cli` argv it runs. Each NAME runs in a FRESH child process whose environment IS the dump, so every
import-time mutation of `import gpuwrf` (release fast-path defaults, x64, command-buffer / autotune XLA_FLAGS, ...) happens
as in the measured process. The child imports gpuwrf.cli (backend-dark) and reports
gpuwrf.runtime.aot_cheap_key.global_trace_env_hash(), the product gpu_allocator.configure_cli_pool plan key and whether
that plan reads back valid (gpu_allocator.read_plan), and whether the JAX backend stayed uninitialized.
  real mode: the dump's own JAX_PLATFORMS (cuda), exactly the product launcher's sizing child (scripts/parallel_cases.py
            _size_here/product_sizing): XLA_PYTHON_CLIENT_PREALLOCATE=false guard in the child env, allocator 'platform'
            in the probe dict (the pool step stops right after the key: no device or NVML query), SystemExit if any JAX
            backend got initialized -> exit 0 iff every NAME's plan reads back valid, the child stayed backend-dark, and
            names sharing a GROUP prefix ('<group>:<label>') agree on trace hash + plan key.
  --emulate-cpu (dry run, no lock): JAX_PLATFORMS=cpu and gpuwrf.backends.platform.target_backend preloaded as 'cuda';
            platform-pinned import config differs from the real run, so only the GROUP agreement + darkness are gated.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

CHILD = r'''
import json, os, sys
emulate, plan_dir, argv = sys.argv[1] == "1", sys.argv[2], json.loads(sys.argv[3])
if emulate:
    import importlib.util
    root = next(p for p in sys.path if os.path.isfile(os.path.join(p, "gpuwrf", "backends", "platform.py")))
    spec = importlib.util.spec_from_file_location("gpuwrf.backends.platform", os.path.join(root, "gpuwrf/backends/platform.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.target_backend = lambda environ=None: "cuda"
    sys.modules["gpuwrf.backends.platform"] = mod
from gpuwrf import cli
from gpuwrf.runtime import aot_cheap_key, gpu_allocator
from jax._src import xla_bridge
gpu_allocator._pool_device_memory = lambda: (_ for _ in ()).throw(AssertionError("no NVML/device query in keycheck"))
probe = dict(os.environ)
probe.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)   # the guard is not part of the arm env (only stops a stray backend)
probe["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"   # stop right after the plan key (as parallel_cases._size_here)
trace = aot_cheap_key.global_trace_env_hash()
args = cli.build_parser().parse_args(argv)
gpu_allocator.configure_cli_pool(args, probe)
session = gpu_allocator._SESSION or {}
key = session.get("key")
plan, err = None, None
if key:
    try:
        plan = gpu_allocator.read_plan(session["path"], key, session["domains"])
    except Exception as exc:
        err = repr(exc)[:300]
if getattr(xla_bridge, "_backends", None):
    raise SystemExit("keycheck initialised a JAX backend")
print("KEYCHECK " + json.dumps({
    "trace_env_hash": trace, "plan_key": key, "plan_path": str(session.get("path")), "plan_read_error": err,
    "plan_found": plan is not None and os.path.dirname(str(session.get("path"))) == plan_dir.rstrip("/"),
    "plan_budget_bytes": (plan or {}).get("budget_bytes"),
    "xla_flags": os.environ.get("XLA_FLAGS", ""), "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
    "hashed_gpuwrf_env": sorted(k for k in os.environ if k.startswith("GPUWRF_") and not aot_cheap_key.trace_env_is_inert(k)),
    "source_file": gpu_allocator.__file__, "backend_dark": not xla_bridge.backends_are_initialized()}))
'''

args = sys.argv[1:]
emulate = args[0] == "--emulate-cpu"
if emulate:
    args = args[1:]
plan_dir, out = Path(args[0]), Path(args[1])
res = {}
for spec in args[2:]:
    name, dump, argv = spec.split("=", 2)
    env = dict(kv.split("=", 1) for kv in Path(dump).read_bytes().decode().split("\0") if "=" in kv)
    if emulate:
        env["JAX_PLATFORMS"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # belt and braces (product_sizing): the import never inits a backend
    r = subprocess.run([sys.executable, "-c", CHILD, "1" if emulate else "0", str(plan_dir), argv], env=env,
                       capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("KEYCHECK ")]
    res[name] = json.loads(line[-1][9:]) if line else {"error": f"rc {r.returncode}: {r.stderr[-1500:]}"}
    res[name]["argv"] = json.loads(argv)
    res[name]["cli_stderr"] = [l for l in r.stderr.splitlines() if l.startswith("gpuwrf: ")][:3]
groups = {}
for name, r in res.items():
    groups.setdefault(name.split(":", 1)[0], []).append(r)
agree = {g: len({(r.get("trace_env_hash"), r.get("plan_key")) for r in rs}) == 1 and all("error" not in r for r in rs)
         for g, rs in groups.items()}
dark = all(r.get("backend_dark") is True for r in res.values())
found = all(r.get("plan_found") for r in res.values())
ok = all(agree.values()) and dark and (emulate or found)
out.write_text(json.dumps({"ok": ok, "mode": "emulate-cpu" if emulate else "real", "groups_agree": agree,
                           "backend_dark": dark, "plans_found": found, "arms": res}, indent=1) + "\n")
print(json.dumps({"ok": ok, "mode": "emulate-cpu" if emulate else "real", "groups_agree": agree, "backend_dark": dark,
                  "plan": {k: [(v.get("plan_key") or "")[:12], v.get("plan_found")] for k, v in res.items()},
                  "trace": {k: (v.get("trace_env_hash") or "")[:12] for k, v in res.items()}}))
sys.exit(0 if ok else 1)
