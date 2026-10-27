"""Root-level mechanism of the ON-default class (manager 04:05Z decision (b)), CPU only.

One process per arm on the Swiss single-domain case, one nested root step, release defaults,
no AOT/JAX cache (fresh trace per process):
  off    no surface controls;
  on     the nine W2 controls at their prior values, traced operands (product path);
  const  identical ON path, but ``traced_controls`` returns the SAME prior values as
         compile-time constants (LS10/LS43 intervention at root level).
``--census`` (separate runs, all arms identical instrumentation) adds, per column, the number of
canopy Newton sweeps (VEGE_FLUX loop1, WRF :3877/:4075-4080) after which the column was still
active, and the |DTV| of the last executed sweep; carried out of the column kernel as one extra
output and sent to the host with jax.debug.callback (instrumented runs are compared only with
instrumented runs; the instrumentation is not product code).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

def patch_vege_source(source):
    """Census instrumentation of VEGE_FLUX loop1 (sweep count + last |DTV|); pure text patch."""
    old_loop = """    if in_column_kernel():  # one sweep body inside the Triton kernel (#14)
        carry = lax.fori_loop(2, NITERC + 1, sweep, sweep(1, carry))
    else:
        for it in range(1, NITERC + 1):
            carry = sweep(it, carry)
"""
    new_loop = """    def _w2_counted(it, cc):
        c, n, d = cc
        prev_tv, was_active = c[0][22], c[0][-1]   # carry slots: tv, active (before this sweep)
        c = sweep(it, c)
        return (c, n + c[0][-1].astype(prev_tv.dtype), jnp.where(was_active, jnp.abs(c[0][22] - prev_tv), d))
    _w2_cc = (sweep(1, carry), jnp.zeros_like(tah), jnp.zeros_like(tah))
    if in_column_kernel():  # one sweep body inside the Triton kernel (#14)
        carry, _w2_n, _w2_d = lax.fori_loop(2, NITERC + 1, _w2_counted, _w2_cc)
    else:
        for it in range(2, NITERC + 1):
            _w2_cc = _w2_counted(it, _w2_cc)
        carry, _w2_n, _w2_d = _w2_cc
"""
    assert source.count(old_loop) == 1, "VEGE_FLUX loop text changed; census instrumentation must be updated"
    source = source.replace(old_loop, new_loop)
    old_ret = '        "t2mv": t2mv, "q2v": q2v,\n'
    assert source.count(old_ret) == 1
    source = source.replace(old_ret, old_ret + '        "w2_active_sweeps": _w2_n, "w2_last_dtv": _w2_d,\n')
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--arm", choices=("off", "on", "const"), required=True)
    p.add_argument("--census", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu" and os.environ.get("GPUWRF_FAST_DEFAULTS") == "1"
    assert os.environ.get("GPUWRF_NESTED_AOT") == "0" and os.environ.get("GPUWRF_JAX_CACHE") == "0"
    sys.path.insert(0, str(a.repo / "src"))
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
    from gpuwrf.physics.noahmp import energy  # noqa: E402
    from gpuwrf.runtime import domain_tree as dt  # noqa: E402
    from gpuwrf.surface_params import controls as sc  # noqa: E402

    assert jax.devices()[0].platform == "cpu"
    census = {}
    if a.census:
        source = patch_vege_source(textwrap.dedent(inspect.getsource(energy._vege_flux)))
        namespace = dict(energy.__dict__)
        exec(compile(source, energy.__file__ + ":w2_census", "exec"), namespace)
        energy._vege_flux = namespace["_vege_flux"]
        original_call = energy.column_call

        def _save(name):
            def save(value):
                census[name] = np.asarray(value).copy()
            return save

        def column_call(fn, args, **kw):
            result = original_call(fn, args, **kw)
            if isinstance(result, tuple) and isinstance(result[0], dict) and "w2_active_sweeps" in result[0]:
                vf = dict(result[0])
                for key in ("w2_active_sweeps", "w2_last_dtv"):
                    jax.debug.callback(_save(key), vf.pop(key))
                result = (vf, *result[1:])
            return result
        energy.column_call = column_call

    cfg = NestedPipelineConfig(a.case, a.out.parent / "unused-output", a.out.parent / "unused-proof", hours=3, max_dom=1)
    hierarchy, bundles, meta, start, steps, initial = _load_domains(cfg, ("d01",))
    names = sorted(set(sc.LSM_CONTROLS) - {"rsmin"})
    receipt = {"arm": a.arm, "census": a.census, "repo": str(a.repo), "case": str(a.case), "names": names,
               "git_head": os.popen(f"git -C {a.repo} rev-parse HEAD").read().strip(),
               "env": {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_NESTED_AOT",
                                                       "GPUWRF_JAX_CACHE", "JAX_ENABLE_COMPILATION_CACHE")}}
    if a.arm != "off":
        values = {name: sc.prior_controls(b.namelist, names, land_state=initial[name].noahmp_land)
                  for name, b in bundles.items()}
        bundles = {name: dataclasses.replace(b, namelist=sc.bind_domain_controls(
            b.namelist, values, name, land_state=initial[name].noahmp_land)) for name, b in bundles.items()}
        receipt["controls_digest"] = sc.controls_digest(values["d01"])
        if a.arm == "const":
            host = {k: np.asarray(v).copy() for k, v in values["d01"].items()}
            traced = sc.traced_controls

            def constant_controls(namelist, clock_base):
                real = traced(namelist, clock_base)
                if real is None:
                    return None
                assert set(real) == set(host)
                # Same values and dtypes, but compile-time constants instead of clock operands.
                return {k: jnp.asarray(host[k], dtype=real[k].dtype) for k in real}
            sc.traced_controls = constant_controls
    tree = dt.DomainTree.from_domains(hierarchy, bundles)
    runtime = dt._prepare_operational_domain_tree_runtime(tree)
    result = dt.run_operational_domain_tree(tree, root_steps=1, carries=initial, initial_own_steps={"d01": 0},
                                            prepared_runtime=runtime, max_event_tail=0)
    jax.block_until_ready(result.carries)
    jax.effects_barrier()
    leaves = {jax.tree_util.keystr(path): np.asarray(value)
              for path, value in jax.tree_util.tree_flatten_with_path(result.carries)[0]}
    land = result.carries["d01"].noahmp_land
    for field in ("tv", "tg", "tah", "eah"):
        leaves["NAMED.noahmp_land." + field] = np.asarray(getattr(land, field))
    for key, value in census.items():
        leaves["CENSUS." + key] = value
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **{f"L{i:04d}": v for i, v in enumerate(leaves.values())})
    receipt["leaf_names"] = list(leaves)
    receipt["leaf_sha256"] = [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in leaves.values()]
    receipt["census_keys"] = sorted(census)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"arm": a.arm, "census": a.census, "leaves": len(leaves), "census_keys": sorted(census)}), flush=True)


if __name__ == "__main__":
    main()
