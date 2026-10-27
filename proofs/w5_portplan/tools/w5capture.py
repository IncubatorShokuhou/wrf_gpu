"""pytest plugin (w5-portplan W5P01): capture every Pallas kernel's Triton IR during CPU tests.

CPU only. For each distinct pallas_call the test executes (interpreted, unchanged), it re-traces the
call with abstract inputs, takes the pallas_call eqn and lowers its kernel jaxpr with JAX's own Triton
lowering for platform 'rocm' (and 'cuda'/120 as control), writing TTIR text + metadata. No test logic
or numerics are changed; capture failures are recorded, never raised.
"""
import hashlib, inspect, json, os, re, threading, traceback

OUT = os.environ.get("W5_CAPTURE_DIR")
_tls = threading.local()
_seen = set()


def _site():
    for fr in inspect.stack()[2:]:
        fn = fr.filename
        if "/src/gpuwrf/" in fn and "backends/pallas.py" not in fn:
            return fn.split("/src/gpuwrf/")[1] + ":" + str(fr.lineno), fr.function
    return "?", "?"


def _record(meta, name, text=None):
    path = os.path.join(OUT, name)
    if text is not None:
        with open(path, "w") as f:
            f.write(text)
    with open(os.path.join(OUT, "index.jsonl"), "a") as f:
        f.write(json.dumps(meta) + "\n")


def pytest_configure(config):
    if not OUT:
        return
    os.makedirs(OUT, exist_ok=True)
    import jax
    from jax.experimental import pallas as jp
    from jax._src.pallas.triton import lowering as tl
    import gpuwrf.backends.pallas as facade

    orig = jp.pallas_call
    orig_facade = facade._jax_call

    def facade_call(*a, **kw):
        _tls.cp = kw.get("compiler_params")
        try:
            return orig_facade(*a, **kw)
        finally:
            _tls.cp = None

    def capture(kernel, a, kw, cp, args, site):
        kw2 = dict(kw); kw2.pop("compiler_params", None); kw2["interpret"] = True
        specs = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, "shape") else x, args)
        jaxpr = jax.make_jaxpr(orig(kernel, *a, **kw2))(*specs)
        eqns = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "pallas_call"]
        for e in eqns:
            p = e.params
            name = getattr(p["jaxpr"].debug_info, "func_name", "kernel")
            gm = p["grid_mapping"]
            meta = {"site": site[0], "function": site[1], "kernel": name, "test": os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0],
                    "grid": [int(g) if isinstance(g, int) else str(g) for g in gm.grid],
                    "in_avals": [f"{v.aval.dtype}{list(v.aval.shape)}" for v in e.invars][:64],
                    "n_in": len(e.invars), "n_out": len(e.outvars),
                    "num_warps": getattr(cp, "num_warps", None), "num_stages": getattr(cp, "num_stages", None),
                    "compiler_params_type": type(cp).__name__ if cp is not None else None}
            for plat, cc in (("rocm", None), ("cuda", 120)):
                try:
                    res = tl.lower_jaxpr_to_triton_module(p["jaxpr"], gm, plat, cc)
                    text = res.module.operation.get_asm(enable_debug_info=False)
                    h = hashlib.sha256(re.sub(r"loc\([^)]*\)", "", text).encode()).hexdigest()[:16]
                    key = (plat, h)
                    m = dict(meta, platform=plat, ttir_sha16=h, ttir_chars=len(text), status="LOWERED",
                             lowered_grid=list(res.grid) if hasattr(res, "grid") else None)
                    if key in _seen:
                        m["status"] = "DUPLICATE"; _record(m, None) if False else _record(m, "/dev/null")
                        continue
                    _seen.add(key)
                    _record(m, f"{plat}_{h}.ttir", text)
                except Exception as err:  # lowering failure is a result
                    _record(dict(meta, platform=plat, status="LOWER_FAIL", error=f"{type(err).__name__}: {str(err)[:400]}"), "/dev/null")

    def wrap(kernel, a, kw, cp):
        site = _site()
        f = orig(kernel, *a, **kw)
        def g(*args):
            try:
                capture(kernel, a, kw, cp, args, site)
            except Exception as err:
                _record({"site": site[0], "function": site[1], "status": "CAPTURE_FAIL", "error": f"{type(err).__name__}: {str(err)[:400]}",
                         "trace": traceback.format_exc()[-800:]}, "/dev/null")
            if os.environ.get("W5_CAPTURE_FAST") == "1":  # shapes only: skip interpreted execution
                import jax.numpy as jnp
                return jax.tree_util.tree_map(lambda o: jnp.zeros(o.shape, o.dtype), kw.get("out_shape", a[0] if a else None))
            return f(*args)
        return g

    def direct(kernel, *a, **kw):
        cp = getattr(_tls, "cp", None) or kw.get("compiler_params")
        return wrap(kernel, a, kw, cp)

    jp.pallas_call = direct
    facade._jax_call = facade_call
    facade._pallas.pallas_call = direct
