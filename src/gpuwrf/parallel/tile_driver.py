"""Run one domain decomposed over several devices: the unmodified single-domain step per tile + halo exchange.

Single process, one device per tile (MPMD: each tile compiles its own program because the step bakes the
tile's horizontal statics into the executable). Per exchange interval each tile executes ONE fused program
``unpack(incoming halos) -> _advance_chunk_fori(k steps) -> pack(outgoing halos)``. A message is one uint8
vector per (src, dst) pair holding every halo rectangle of every leaf (bit-exact bitcasts), padded to its
round length. Transports: ``device_put`` (PJRT device-to-device copy: CUDA peer copy, or host-staged where
P2P is unavailable, e.g. GeForce RTX 4090) and ``ppermute`` (NCCL collective-permute, one round per
neighbour offset, SPMD over the equal-shaped message vectors only). Dispatch is asynchronous: tiles run
concurrently and nothing synchronizes the host inside an interval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from gpuwrf.parallel.decomposition import DecompositionSpec, Transfer, classify, exchange_plan, gather_leaves, slice_namelist, slice_tree


def required_halo(acoustic_substeps: int, exchange_every: int = 1, *, bdy_width: int = 5,
                  shadow_radius: int = 0) -> int:
    """Candidate dependency bound for the audited release operators (GPU qualification pending):
    per RK stage k: momentum/theta h5 advection incl. mass coupling 4 + acoustic substeps n_k (radius 1 each:
    advance_uv/smdiv, advance_mu_t, advance_w) + post-acoustic scalar PD advection 4; stages n_k = 1, ns/2, ns;
    +1 physics destagger / w_surface_reset. Step radius R = 24 + 1 + ns//2 + ns + 1; each internal tile edge
    re-applies the specified/relaxation zone (bdy_width rows) every step; +1 because a tile's lowest OWNED face
    (x0 / y0) is computed from the halo mass cell x0-1 (staggering). Radiation precedes dynamics, so its
    shadow radius ADDS to R, rather than taking max(R, shadow). See configuration_halo for admission."""
    ns = int(acoustic_substeps)
    r = 24 + 1 + ns // 2 + ns + 1
    return int(exchange_every) * (r + int(shadow_radius)) + int(bdy_width) + 1


def configuration_halo(namelist, exchange_every: int = 1) -> int:
    """Bound for the supported release configuration; refuse unqualified nonlocal operators.

    The topographic scan visits steps 1..int(L/dx+1), with a second interpolation neighbour. In each
    azimuth branch |tan| <= 1, hence max_steps+1 bounds BOTH horizontal axes, also for dx != dy.
    Physics-local columns, h5/h6 advection, PD/FCT and sixth-order diffusion use the base cone.
    Domain-wide mean/limiter paths cannot be represented by a finite halo here.
    """
    if int(getattr(namelist, "km_opt", 0)) in (2, 5) or float(getattr(namelist, "const_nu_m2_s", 0)) > 0:
        raise ValueError("decomposition does not support domain-wide diffusion reductions (km_opt=2/5/const_nu)")
    if not bool(getattr(namelist, "disable_guards", True)):
        raise ValueError("decomposition requires strict guards; the legacy theta limiter has global reductions")
    if hasattr(namelist, "specified_adv_degrade") and not namelist.specified_adv_degrade:
        raise ValueError("decomposition requires specified-boundary advection; global periodic wrapping is unsupported")
    if int(getattr(namelist, "rk_order", 3)) != 3 or int(getattr(namelist, "h_sca_adv_order", 5)) > 6:
        raise ValueError("decomposition dependency cone is qualified only for RK3 and advection order <= 6")
    # New schemes may introduce spatial operators; opt-in scheme wiring is not a halo qualification.
    if bool(getattr(namelist, "run_physics", True)):
        supported = {"mp_physics": (8,), "bl_pbl_physics": (5,), "sf_sfclay_physics": (5,),
                     "sf_surface_physics": (4,), "ra_lw_physics": (4,), "ra_sw_physics": (4,),
                     "cu_physics": (0, 1), "sf_urban_physics": (0,)}
        for name, codes in supported.items():
            if int(getattr(namelist, name, codes[0])) not in codes:
                raise ValueError(f"decomposition dependency cone is not qualified for {name}={getattr(namelist, name)}")
        if hasattr(namelist, "radiation_static") and namelist.radiation_static is None:
            raise ValueError("decomposition requires global radiation statics; tile-index latitude fallback is unsupported")
    shadow = 0
    if int(getattr(namelist, "topo_shading", 0)) == 1 and int(getattr(namelist, "slope_rad", 0)) == 1:
        dx = float(namelist.grid.projection.dx_m)
        length = float(namelist.topo_shadow_length_m)
        if not np.isfinite(dx) or dx <= 0 or not np.isfinite(length) or length < 0:
            raise ValueError("decomposition needs finite positive dx and nonnegative topographic shadow length")
        shadow = int(length / dx + 1.0) + 1
    bdy = getattr(namelist, "boundary_config", None)
    width = max(int(getattr(bdy, "spec_bdy_width", 5)),
                int(getattr(bdy, "spec_zone", 1)) + int(getattr(bdy, "relax_zone", 4)))
    return required_halo(namelist.acoustic_substeps, exchange_every, bdy_width=width, shadow_radius=shadow)


def refuse_unowned_diagnostics(carry) -> None:
    """Reduced tile census values include halos and cannot be gathered as owned-cell diagnostics.

    Refuse explicitly until masks/global indices are threaded into the model's diagnostic operators.
    Never silently disable F2 or sum already reduced tile counters.
    """
    import jax

    from gpuwrf.diagnostics.census import DeviceCensus, enabled

    if enabled() or any(isinstance(x, DeviceCensus) for x in
                        jax.tree_util.tree_leaves(carry, is_leaf=lambda x: isinstance(x, DeviceCensus))):
        raise ValueError("census-enabled decomposition is unsupported: owned-cell guard counts, first events "
                         "and water budgets require global ownership; F2 qualification is unavailable")


def live_input_leaves(step_fn: Callable, carry, *args) -> list[bool]:
    """Which carry leaves ONE step reads (jaxpr dead-code elimination over ``step_fn(carry, *args)`` with every
    output used). A leaf the step overwrites before reading (scratch, diagnostics) needs no halo refresh: its
    halo is recomputed from exchanged leaves. Conservative: every cond branch counts (radiation and ordinary)."""
    import jax
    from jax._src.interpreters import partial_eval as pe

    closed = jax.make_jaxpr(step_fn)(carry, *args)
    _, used = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    return list(used[: len(jax.tree_util.tree_leaves(carry))])


def release_step_live_leaves(carry, namelist, clock_base, cadence: int) -> list[bool]:
    """live_input_leaves of the release step body (operational_mode._physics_boundary_step with the traced
    radiation predicate of _advance_chunk_fori)."""
    import jax.numpy as jnp

    from gpuwrf.runtime import operational_mode as om

    def one(c, si):
        rr = jnp.equal(jnp.mod(si - 1, cadence), 0)
        return om._physics_boundary_step(c, namelist, si, run_radiation=rr, debug=False, clock_base=clock_base)

    return live_input_leaves(one, carry, jnp.asarray(1, dtype=jnp.int32))


def _unalias(tree, *, protected=None):
    """One owner per buffer before the donating tile programs: the product carry holds some arrays at two
    positions, and a full-extent slice / device_put can return the same buffer (PjRt refuses to donate a
    buffer twice). Repeats are copied once (cf. operational_mode.unalias_donated_carry)."""
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    def buffer_key(leaf):
        try:
            return tuple((sh.device.id, sh.data.unsafe_buffer_pointer()) for sh in leaf.addressable_shards)
        except Exception:  # noqa: BLE001 - deleted/abstract buffers: identity is the best we have
            return ("id", id(leaf))

    # Full-extent slices/device_put may alias the caller's global carry too. It must survive tile
    # donation for output-boundary scatter and symmetric benchmark warmup resets. Copy only aliases,
    # at setup/scatter, rather than adding a full-tile copy inside each advance interval.
    seen = {buffer_key(x) for x in jax.tree_util.tree_leaves(protected) if isinstance(x, jax.Array)}
    changed = False
    for i, leaf in enumerate(leaves):
        if not isinstance(leaf, jax.Array):
            continue
        key = buffer_key(leaf)
        if key in seen:
            leaves[i] = jnp.copy(leaf)
            changed = True
        else:
            seen.add(key)
    return jax.tree_util.tree_unflatten(treedef, leaves) if changed else tree


def _nbytes(m: Transfer, leaf) -> int:
    lead = int(np.prod(np.shape(leaf)[:-2], dtype=np.int64))
    return lead * (m.src_y[1] - m.src_y[0]) * (m.src_x[1] - m.src_x[0]) * np.dtype(leaf.dtype).itemsize


def _to_bytes(x):
    import jax.numpy as jnp
    from jax import lax

    if x.dtype == jnp.bool_:
        x = x.astype(jnp.uint8)
    flat = x.reshape(-1)
    if flat.dtype.itemsize == 1:
        return lax.bitcast_convert_type(flat, jnp.uint8)
    return lax.bitcast_convert_type(flat, jnp.uint8).reshape(-1)


def _from_bytes(b, dtype, n: int):
    import jax.numpy as jnp
    from jax import lax

    dt = np.dtype(dtype)
    target = jnp.uint8 if dt == np.bool_ else dt
    size = np.dtype(target).itemsize
    out = lax.bitcast_convert_type(b, target) if size == 1 else lax.bitcast_convert_type(b.reshape(n, size), target)
    return out.astype(jnp.bool_) if dt == np.bool_ else out


def _pack(leaves: list, moves: list[Transfer], length: int):
    """All rectangles of one message as ONE uint8 vector of ``length`` bytes (zero padded)."""
    import jax.numpy as jnp

    parts = [_to_bytes(leaves[m.leaf][..., m.src_y[0]:m.src_y[1], m.src_x[0]:m.src_x[1]]) for m in moves]
    used = sum(int(p.shape[0]) for p in parts)
    if length > used:
        parts.append(jnp.zeros((length - used,), dtype=jnp.uint8))
    return jnp.concatenate(parts) if len(parts) > 1 else parts[0]


def _unpack(leaves: list, msg, moves: list[Transfer]) -> list:
    leaves = list(leaves)
    off = 0
    for m in moves:
        leaf = leaves[m.leaf]
        shape = tuple(leaf.shape[:-2]) + (m.dst_y[1] - m.dst_y[0], m.dst_x[1] - m.dst_x[0])
        n = int(np.prod(shape, dtype=np.int64))
        nb = n * np.dtype(leaf.dtype).itemsize
        vals = _from_bytes(msg[off:off + nb], leaf.dtype, n).reshape(shape)
        leaves[m.leaf] = leaf.at[..., m.dst_y[0]:m.dst_y[1], m.dst_x[0]:m.dst_x[1]].set(vals)
        off += nb
    return leaves


def message_rounds(spec: DecompositionSpec, pairs) -> list[list[tuple[int, int]]]:
    """Pairs grouped by tile offset (dix, diy): within one offset every rank sends <= 1 and receives <= 1
    message, so each group is one collective-permute round."""
    rounds: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for s, d in sorted(pairs):
        off = (spec.tiles[d].ix - spec.tiles[s].ix, spec.tiles[d].iy - spec.tiles[s].iy)
        rounds.setdefault(off, []).append((s, d))
    return [rounds[k] for k in sorted(rounds)]


class DevicePutTransport:
    """PJRT device-to-device copies (host-staged by the runtime where GPU peer access is unavailable)."""

    name = "device_put"

    def __init__(self, devices, rounds, lengths):
        self.devices = devices

    def route(self, outboxes: dict, inbox: list) -> None:
        import jax

        for src, out in outboxes.items():
            for dst, msg in out.items():
                inbox[dst][src] = jax.device_put(msg, self.devices[dst])


class PpermuteTransport:
    """NCCL (GPU) / XLA (CPU) collective-permute in ONE executable per exchange.

    Launching a separate asynchronous executable for every neighbour offset lets CPU ranks enter
    different rendezvous while earlier rank work is still queued. A single ordered collective program
    gives every rank the same rendezvous sequence, and reduces host launches on GPUs.
    """

    name = "ppermute"

    def __init__(self, devices, rounds, lengths):
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        self.n = len(devices)
        self.devices = devices
        mesh = Mesh(np.array(devices, dtype=object), ("t",))
        self.sharding = NamedSharding(mesh, P("t"))
        self.rounds = []
        for perm in rounds:
            length = lengths[perm[0]]
            idle = [jax.device_put(np.zeros((1, length), np.uint8), d) for d in devices]
            self.rounds.append((perm, length, idle))
        permutations = tuple(tuple(perm) for perm in rounds)

        def exchange(*messages):
            return tuple(jax.lax.ppermute(msg, "t", perm) for msg, perm in zip(messages, permutations))

        specs = tuple(P("t") for _ in rounds)
        self.exchange = jax.jit(jax.shard_map(exchange, mesh=mesh, in_specs=specs, out_specs=specs))

    def route(self, outboxes: dict, inbox: list) -> None:
        import jax

        inputs = []
        for perm, length, idle in self.rounds:
            sends = {s: d for s, d in perm}
            pieces = [outboxes[r][sends[r]].reshape(1, length) if r in sends else idle[r] for r in range(self.n)]
            inputs.append(jax.make_array_from_single_device_arrays((self.n, length), self.sharding, pieces))
        results = self.exchange(*inputs) if inputs else ()
        for (perm, length, _idle), got in zip(self.rounds, results):
            by_dev = {sh.device: sh.data for sh in got.addressable_shards}
            for s, d in perm:
                inbox[d][s] = by_dev[self.devices[d]].reshape(length)


TRANSPORTS = {"device_put": DevicePutTransport, "ppermute": PpermuteTransport}


@dataclass
class TileRuntime:
    """Decomposed run state. Build with :meth:`create`; advance with :meth:`advance`; :meth:`gather` -> global."""

    spec: DecompositionSpec
    devices: list
    kinds: list[tuple]
    treedef: Any
    template: list  # per leaf: host copy (non-h leaves) or ShapeDtypeStruct (h leaves) for gather/bytes
    namelists: list
    carries: list
    clock_bases: list
    steps: list  # per-tile jitted fused programs
    plan: dict  # (src, dst) -> [Transfer]
    lengths: dict  # (src, dst) -> message bytes incl. round padding
    payload: dict  # (src, dst) -> message bytes without padding
    transport: Any
    inbox: list  # per dst: {src: message}
    exchange_every: int
    cadence: int
    timings: dict = field(default_factory=dict)
    starts: list = field(default_factory=list)
    trip_counts: list = field(default_factory=list)
    cadences: list = field(default_factory=list)
    next_step: int = 1
    packers: list = field(default_factory=list)

    @classmethod
    def create(cls, carry, namelist, spec: DecompositionSpec, devices: list, *, cadence: int, exchange_every: int = 1,
               clock_base=None, advance: Callable | None = None, plan_filter: Callable | None = None,
               transport: str = "device_put", live: list[bool] | None = None, start_step: int = 1) -> "TileRuntime":
        """``advance(carry, namelist, start_step, clock_base, *, n_steps, cadence)`` defaults to the release
        ``operational_mode._advance_chunk_fori``. ``plan_filter`` (tests: halo mutants) may edit the plan.
        ``live``: per carry leaf, whether its halo is exchanged (default all; see release_step_live_leaves)."""
        import jax

        refuse_unowned_diagnostics(carry)
        n = len(spec.tiles)
        if len(devices) < n:
            raise ValueError(f"{n} tiles need {n} devices, got {len(devices)}")
        devices = list(devices[:n])
        if advance is None or clock_base is None:
            from gpuwrf.runtime import operational_mode as om

            advance = advance or om._advance_chunk_fori
            clock_base = om.build_clock_base(namelist) if clock_base is None else clock_base
        leaves, treedef = jax.tree_util.tree_flatten(carry)
        kinds = [classify(np.shape(x), spec) for x in leaves]
        # host copies only of the leaves gather() takes from here (static strips/flat, replicated); shapes otherwise
        template = [jax.ShapeDtypeStruct(np.shape(x), x.dtype) if kd[0] == "h" else np.asarray(x)
                    for x, kd in zip(leaves, kinds)]
        if live is not None and len(live) != len(kinds):
            raise ValueError("live mask length differs from the carry leaf count")
        plan = exchange_plan(spec, [k if live is None or live[i] else ("dead",) for i, k in enumerate(kinds)])
        if plan_filter is not None:
            plan = plan_filter(plan)
        payload = {pair: sum(_nbytes(m, template[m.leaf]) for m in moves) for pair, moves in plan.items()}
        rounds = message_rounds(spec, plan)
        lengths = {}
        for perm in rounds:
            longest = max(payload[p] for p in perm)
            for p in perm:
                lengths[p] = longest
        namelists, carries, clocks, steps = [], [], [], []
        k = int(exchange_every)
        for t in spec.tiles:
            dev = devices[t.rank]
            ins = [(src, plan[(src, dst)]) for (src, dst) in sorted(plan) if dst == t.rank]
            outs = [(dst, plan[(src, dst)], lengths[(src, dst)]) for (src, dst) in sorted(plan) if src == t.rank]

            def step_fn(c, nl, start, clk, inbox, n_steps, cad, _ins=ins, _outs=outs):
                # n_steps/cadence stay TRACED (as in the release CLI call of _advance_chunk_fori): a constant trip
                # count lets XLA specialise the step loop and changes round-off vs the single-domain program.
                lv, td = jax.tree_util.tree_flatten(c)
                for src, moves in _ins:
                    lv = _unpack(lv, inbox[src], moves)
                c = advance(jax.tree_util.tree_unflatten(td, lv), nl, start, clk, n_steps=n_steps, cadence=cad)
                lv = jax.tree_util.tree_leaves(c)
                return c, {dst: _pack(lv, moves, length) for dst, moves, length in _outs}, start + n_steps

            namelists.append(jax.device_put(slice_namelist(namelist, t, spec), dev))
            carries.append(_unalias(jax.device_put(slice_tree(carry, t, spec), dev), protected=carry))
            clocks.append(jax.device_put(clock_base, dev))
            steps.append(jax.jit(step_fn, donate_argnums=(0,)))
        rt = cls(spec, devices, kinds, treedef, template, namelists, carries, clocks, steps, plan, lengths, payload,
                 TRANSPORTS[transport](devices, rounds, lengths), [dict() for _ in spec.tiles], k, int(cadence))
        rt._prepare_controls(start_step)
        rt._initial_halos()
        return rt

    def _prepare_controls(self, start_step: int) -> None:
        """Place traced control scalars once, outside advance/timing; each tile advances its own clock."""
        import jax

        self.starts = [jax.device_put(np.int32(start_step), d) for d in self.devices]
        self.trip_counts = [jax.device_put(np.int32(self.exchange_every), d) for d in self.devices]
        self.cadences = [jax.device_put(np.int32(self.cadence), d) for d in self.devices]
        self.next_step = int(start_step)

    def _initial_halos(self) -> None:
        """First messages packed from the freshly sliced tiles (their halos are already exact)."""
        import jax

        if not self.packers:
            for t in self.spec.tiles:
                outs = [(dst, self.plan[(src, dst)], self.lengths[(src, dst)]) for (src, dst) in sorted(self.plan) if src == t.rank]
                self.packers.append(jax.jit(lambda c, _outs=outs: {d: _pack(jax.tree_util.tree_leaves(c), m, n)
                                                                for d, m, n in _outs}))
        outboxes = {t.rank: self.packers[t.rank](self.carries[t.rank]) for t in self.spec.tiles}
        self.transport.route(outboxes, self.inbox)

    def advance(self, start_step: int, n_steps: int) -> None:
        """Advance every tile by ``n_steps`` (a multiple of exchange_every) from 1-based ``start_step``."""
        if n_steps < 0 or n_steps % self.exchange_every:
            raise ValueError("n_steps must be a multiple of exchange_every")
        if int(start_step) != self.next_step:
            raise ValueError(f"noncontiguous tile clock: expected {self.next_step}, got {start_step}; "
                             "set start_step at create/scatter, outside the timed loop")
        t0 = time.perf_counter()
        for i in range(0, n_steps, self.exchange_every):
            outboxes = {}
            for t in self.spec.tiles:
                r = t.rank
                self.carries[r], outboxes[r], self.starts[r] = self.steps[r](
                    self.carries[r], self.namelists[r], self.starts[r], self.clock_bases[r], self.inbox[r],
                    self.trip_counts[r], self.cadences[r])
            self.transport.route(outboxes, self.inbox)
        self.next_step += int(n_steps)
        self.timings.setdefault("dispatch_s", []).append(time.perf_counter() - t0)

    def block(self) -> None:
        import jax

        jax.block_until_ready((self.carries, self.inbox, self.starts))

    def scatter(self, carry, *, start_step: int | None = None) -> None:
        """Re-slice a (possibly modified) GLOBAL carry into the tiles, keeping the compiled tile programs (same
        treedef/shapes required); then send fresh initial halos. Used between output intervals, when host-side
        callbacks (output, accumulator resets, lower-boundary updates) may have changed the global carry."""
        import jax

        refuse_unowned_diagnostics(carry)
        leaves, treedef = jax.tree_util.tree_flatten(carry)
        if treedef != self.treedef or [np.shape(x) for x in leaves] != [x.shape for x in self.template]:
            raise ValueError("scatter: carry structure/shapes differ from the decomposed run's carry")
        for i, kind in enumerate(self.kinds):  # static leaves (strips) come from the new carry too
            if kind[0] != "h":
                self.template[i] = np.asarray(leaves[i])
        for t in self.spec.tiles:
            self.carries[t.rank] = _unalias(jax.device_put(slice_tree(carry, t, self.spec), self.devices[t.rank]),
                                          protected=carry)
        if start_step is not None:
            self._prepare_controls(start_step)
        self._initial_halos()

    def close(self) -> None:
        """Release this run's executable references and device buffers (never clear unrelated JAX caches)."""
        for step in (*self.steps, *self.packers):
            step.clear_cache()
        self.steps.clear()
        self.packers.clear()
        self.carries.clear()
        self.inbox.clear()
        self.namelists.clear()
        self.clock_bases.clear()
        self.starts.clear()
        self.trip_counts.clear()
        self.cadences.clear()
        self.transport = None

    def gather_to(self, device):
        """The global carry as device arrays on ``device`` (host assembly, one transfer per leaf)."""
        import jax

        return jax.device_put(self.gather(), device)

    def gather(self):
        """The global carry (host NumPy leaves) from the tiles' owned columns."""
        import jax

        tile_leaves = [[np.asarray(x) for x in jax.tree_util.tree_leaves(c)] for c in self.carries]
        return jax.tree_util.tree_unflatten(self.treedef, gather_leaves(tile_leaves, self.template, self.kinds, self.spec))

    def message_bytes(self) -> dict:
        """Payload bytes per (src, dst) halo message per exchange."""
        return dict(self.payload)


__all__ = ["TRANSPORTS", "TileRuntime", "configuration_halo", "live_input_leaves", "message_rounds",
           "refuse_unowned_diagnostics", "release_step_live_leaves", "required_halo"]
