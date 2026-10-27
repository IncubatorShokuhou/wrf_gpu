"""CPU launch smoke entered from the actual capture child in product CUDA env.

Only device/backend/profiler APIs are substituted. Actual immutable-source
Thompson code executes on one real PROD rain column. This is launch coverage,
not GPU, forecast, speed, byte-identity or WRF-fidelity qualification.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ENV_KEYS=("JAX_PLATFORMS","JAX_ENABLE_X64","GPUWRF_JAX_CACHE","GPUWRF_JAX_CACHE_DIR",
          "JAX_COMPILATION_CACHE_DIR","GPUWRF_XLA_AUTOTUNE_CACHE_DIR","TRITON_CACHE_DIR",
          "CUDA_CACHE_PATH","XLA_PYTHON_CLIENT_ALLOCATOR","XLA_PYTHON_CLIENT_PREALLOCATE",
          "XLA_CLIENT_MEM_FRACTION","BENCH_SRC_REPO","PYTHONPATH","W44_POOL",
          "W44_CAPTURE_ROOTS","W44_CAPTURE_START","W44_HOST_PROFILE","W44_INIT_OBSERVER")
FLAG_KEYS=("GPUWRF_FAST_DEFAULTS","GPUWRF_FP64_DEFAULT","GPUWRF_THOMPSON_FULL_COLUMN",
           "GPUWRF_THOMPSON_NATIVE_REAL","GPUWRF_THOMPSON_MIXED_PHASE_WRF","GPUWRF_CARRY_DONATE")


def main(h):
    assert os.environ.get("W44_LAUNCH_SMOKE")=="1"
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD","0")!="1","CPU smoke must not impersonate an official GPU holder"
    product={k:os.environ.get(k) for k in ENV_KEYS}
    assert product["JAX_PLATFORMS"] in (None,"cuda"),product["JAX_PLATFORMS"]
    assert product["GPUWRF_JAX_CACHE"]=="1"
    assert product["XLA_PYTHON_CLIENT_ALLOCATOR"]=="cuda_async"
    assert product["XLA_PYTHON_CLIENT_PREALLOCATE"]=="true" and product["XLA_CLIENT_MEM_FRACTION"]=="0.25"
    out=Path(sys.argv[sys.argv.index("--out")+1]);out.mkdir(parents=True,exist_ok=True)
    rec={"schema":"w44-perf.launch-smoke.child.v1","product_environment":product,
         "argv":sys.argv,"gpu_executed":False,"model_reached":False,"scope":__doc__}
    if os.environ.get('W44_CPU_PROBE')=='1':
        group=Path('/proc/self/cgroup').read_text().strip()
        relative=group.split(':',2)[2]
        limit=Path('/sys/fs/cgroup')/relative.lstrip('/')/'memory.max'
        maximum=limit.read_text().strip()
        assert maximum!='max' and int(maximum)<=12*1024**3,(group,maximum)
        rec['actual_ram_scope']={'cgroup':relative,'memory_max_bytes':int(maximum)}
    def save():
        (out/"LAUNCH_SMOKE_MODEL.json").write_text(json.dumps(rec,indent=2,allow_nan=False)+"\n")
    save()
    # Capture the product environment before changing only the device layer.
    # Leave JAX_PLATFORMS=cuda in the child's environment; no source flag changes.
    import jax
    from jax._src import xla_bridge
    assert not xla_bridge.backends_are_initialized(),"backend initialized before CPU device substitution"
    jax.config.update("jax_platforms","cpu")
    jax.config.update("jax_enable_compilation_cache",False)  # no CPU executables in paid GPU namespaces
    original_devices=jax.devices
    assert all(d.platform=="cpu" for d in original_devices("cpu"))
    original_profiler=h._cuda_profiler
    boundary_calls=[]
    ran=False

    def boundary(*args,**kwargs):
        nonlocal ran
        jax.devices=original_devices
        devices=original_devices("cpu")
        assert devices and all(d.platform=="cpu" for d in devices)
        rec["device_substitution"]={"jax_platforms_config":jax.config.jax_platforms,
                                    "environment_still":os.environ.get("JAX_PLATFORMS"),
                                    "physical_devices":[d.platform for d in devices]}
        if not ran:
            ran=True
            import numpy as np
            import jax.numpy as jnp
            from gpuwrf.physics import thompson_column as tc
            from gpuwrf.kernels import phys_thompson_full as full
            jax.config.update("jax_enable_compilation_cache",False)
            fixture=Path(__file__).resolve().parents[2]/"tests/v025/b_thompson/fixtures/exit_columns.npz"
            with np.load(fixture) as arrays:
                state=tc.ThompsonColumnState(**{key:jnp.asarray(arrays[key][:1]) for key in tc.ThompsonColumnState.__slots__})
            assert state.qv.shape==(1,44),state.qv.shape
            rec["resolved_product_flags"]={k:os.environ.get(k) for k in FLAG_KEYS}
            h._cuda_profiler=lambda event:boundary_calls.append(event)
            if product["W44_HOST_PROFILE"]=="1":
                import host_observer
                host_observer.install(h)
            h._cuda_profiler("cudaProfilerStart")
            start=time.perf_counter()
            result=full.full_column(state,18.,interpret=True)
            jax.block_until_ready(result)
            rec["compile_and_model_s"]=time.perf_counter()-start
            h._cuda_profiler("cudaProfilerStop")
            leaves=jax.tree.leaves(result)
            assert leaves and all(np.isfinite(np.asarray(x)).all() for x in leaves)
            rec.update(model_reached=True,model_function="gpuwrf.kernels.phys_thompson_full.full_column",
                       model_file_sha256=hashlib.sha256(Path(full.__file__).read_bytes()).hexdigest(),
                       fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),fixture_column=0,
                       fixture_provenance="PROD day d02 original CPU-WRF history 2026-07-26_12, real rain column",
                       input_shape=list(state.qv.shape),dt_s=18.,finite_output_leaves=len(leaves),
                       physical_backend=jax.default_backend(),profiler_device_boundary_events=boundary_calls)
            if product["W44_HOST_PROFILE"]=="1":
                profile=h.REC["w44_host_profile"]
                assert profile["total_calls"]>0
                rec["host_profile"]={k:profile[k] for k in ("start","end","total_calls","total_s")}
            save()
        return devices

    jax.devices=boundary
    try:
        rc=h.main()  # actual CLI parse/pre-exec and genuine GPU guard
    finally:
        jax.devices=original_devices
        h._cuda_profiler=original_profiler
    rec["harness_return_code"]=rc
    # The unchanged production harness must truthfully refuse CPU qualification.
    assert ran and rec["model_reached"] and rc==3,(ran,rc)
    rec["status"]="PASS_CPU_LAUNCH_SMOKE"
    save()
    print(json.dumps({"status":rec["status"],"model_receipt":str(out/"LAUNCH_SMOKE_MODEL.json"),
                      "product_JAX_PLATFORMS":product["JAX_PLATFORMS"],"gpu_executed":False,
                      "model_reached":True,"physical_backend":"cpu","GPU_harness_refusal_rc":rc}))
    return 0

