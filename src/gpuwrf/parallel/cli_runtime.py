"""Opt-in CLI hook (``gpuwrf run --decompose PXxPY`` -> env GPUWRF_DECOMPOSE): advance the root of a native
single-domain run decomposed over the visible devices (experimental).

``domain_tree``'s ``advance`` calls :func:`decomposed_advance` only when the env var is set. The global carry
stays the interface: each call scatters it into the tiles (the compiled tile programs are reused), advances
``n_steps`` with halo exchange every k steps, and gathers the owned columns back onto the carry's device, so
output, accumulator resets, lower-boundary updates and checkpoints see one global carry exactly as before
(restart is partition-independent).
"""

from __future__ import annotations

import os
import re
import weakref
from typing import Any

ENV = "GPUWRF_DECOMPOSE"
# DomainTree is frozen and unhashable. A weak identity side table scopes executables to its lifetime
# without mutating the tree or retaining completed runs. No array hashing/reads on the timestep path.
_RUNS: dict[int, tuple[Any, dict]] = {}
REPORT: dict[str, Any] = {}


def _close_run(tree_id: int) -> None:
    saved = _RUNS.pop(tree_id, None)
    if saved is not None:
        for _token, runtime in saved[1].values():
            runtime.close()
        saved[1].clear()


def parse(value: str) -> dict:
    """'2x1' or '2x1,halo=40,k=1,transport=ppermute,exchange=all' -> config dict (ValueError on anything else)."""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{ENV} is empty")
    m = re.fullmatch(r"(\d+)x(\d+)", parts[0].lower())
    if not m:
        raise ValueError(f"{ENV}: expected PXxPY (e.g. 2x1), got {parts[0]!r}")
    cfg = {"px": int(m.group(1)), "py": int(m.group(2)), "halo": None, "k": 1, "transport": "device_put", "exchange": "live"}
    for p in parts[1:]:
        key, _, val = p.partition("=")
        if key == "halo":
            cfg["halo"] = int(val)
        elif key == "k":
            cfg["k"] = int(val)
        elif key == "transport" and val in ("device_put", "ppermute"):
            cfg["transport"] = val
        elif key == "exchange" and val in ("live", "all"):
            cfg["exchange"] = val
        else:
            raise ValueError(f"{ENV}: unknown option {p!r}")
    if cfg["px"] < 1 or cfg["py"] < 1 or cfg["px"] * cfg["py"] < 2 or cfg["k"] < 1 or (cfg["halo"] is not None and cfg["halo"] < 0):
        raise ValueError(f"{ENV}: need px*py >= 2, k >= 1, halo >= 0")
    return cfg


def decomposed_advance(tree, name: str, carry, start_step: int, n_steps: int, clock_base, *, advance=None) -> Any:
    """``advance`` (tests only) replaces the release step program; the live-leaf analysis then is skipped."""
    import jax

    from gpuwrf.parallel import decomposition as dc
    from gpuwrf.parallel.tile_driver import TileRuntime, configuration_halo, release_step_live_leaves, refuse_unowned_diagnostics

    cfg = parse(os.environ[ENV])
    if len(getattr(tree, "domains", {})) != 1 or getattr(tree, "edges", {}).get(name):
        raise ValueError(f"{ENV}: domain decomposition supports a single domain only ({name} has nests)")
    namelist = tree.domains[name].namelist
    refuse_unowned_diagnostics(carry)
    cadence = int(namelist.radiation_cadence_steps)
    k = cfg["k"]
    if int(n_steps) % k:
        raise ValueError(f"{ENV}: the output interval ({n_steps} steps) must be a multiple of k={k}")
    tree_id = id(tree)
    binding = _RUNS.get(tree_id)
    if binding is None or binding[0]() is not tree:
        _close_run(tree_id)
        binding = (weakref.ref(tree, lambda _ref, _id=tree_id: _close_run(_id)), {})
        _RUNS[tree_id] = binding
    runtimes = binding[1]
    # O(1) object/configuration identity checks, never hash/read grid arrays per step. Namelists are
    # immutable; a new namelist or clock object means a new configuration/run binding.
    # Keep the original binding objects alive: naked ids can be recycled after a short-lived clock
    # container is collected, incorrectly admitting different date/forcing values on a later call.
    token = (namelist, tuple(cfg.items()), clock_base, advance)
    saved = runtimes.get(name)
    same_binding = (saved is not None and saved[0][0] is namelist and saved[0][1] == token[1]
                    and saved[0][2] is clock_base and saved[0][3] is advance)
    rt = saved[1] if same_binding else None
    if rt is not None and rt.next_step != int(start_step):
        # A new forecast may reuse an immutable tree/configuration but restart its step sequence.
        rt = None
    if saved is not None and rt is None:
        saved[1].close()
    if rt is None:
        nproc = cfg["px"] * cfg["py"]
        devices = jax.devices()
        if len(devices) < nproc:
            raise ValueError(f"{ENV}: {nproc} tiles need {nproc} devices, JAX sees {len(devices)}")
        bound = configuration_halo(namelist, k)
        halo = cfg["halo"] if cfg["halo"] is not None else bound
        if advance is None and halo < bound:
            raise ValueError(f"{ENV}: halo={halo} is below the selected configuration dependency bound {bound}")
        spec = dc.DecompositionSpec.build(namelist.grid.nx, namelist.grid.ny, cfg["px"], cfg["py"], halo,
            bdy_width=int(getattr(getattr(namelist, "boundary_config", None), "spec_bdy_width", 5)))
        live = (release_step_live_leaves(carry, namelist, clock_base, cadence)
                if cfg["exchange"] == "live" and advance is None else None)
        rt = TileRuntime.create(carry, namelist, spec, devices[:nproc], cadence=cadence, exchange_every=k,
                                clock_base=clock_base, transport=cfg["transport"], live=live, advance=advance,
                                start_step=int(start_step))
        runtimes[name] = (token, rt)
        REPORT[name] = {"config": cfg, "spec": spec.describe(), "devices": [str(d) for d in devices[:nproc]],
                        "message_bytes": {f"{s}->{d}": b for (s, d), b in rt.message_bytes().items()},
                        "live_leaves": None if live is None else int(sum(live))}
    else:
        rt.scatter(carry)
    rt.advance(int(start_step), int(n_steps))
    target = next(iter(jax.tree_util.tree_leaves(carry)[0].devices()))
    return rt.gather_to(target)


def reset(tree=None) -> None:
    """Release one run's tile cache, or all live run caches (also automatic when its tree is collected)."""
    for tree_id, (ref, _cache) in list(_RUNS.items()):
        if tree is None or ref() is tree:
            _close_run(tree_id)
    REPORT.clear()


__all__ = ["ENV", "REPORT", "decomposed_advance", "parse", "reset"]
