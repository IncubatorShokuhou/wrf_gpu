"""Opt-in regional stream setup, bounded segment flush and durable restart."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from .config import FILENAME, FIELDS, read_streams, region_points


def require_supported_run(input_dir, *, batch_size=1, replay=False):
    path = Path(input_dir) / FILENAME
    if path.is_file():
        specs = read_streams(path)
        if specs and (batch_size != 1 or replay):
            raise ValueError("regional streams require a native B=1 forecast; replay/batching unsupported")


class StreamRuntime:
    def __init__(self, plans, max_root_steps, output_dir, run_start, journal=None, saved=None):
        from .writer import AsyncStreamWriter
        self.plans, self.max_root_steps = plans, max_root_steps
        self.output_dir, self.run_start, self.journal = Path(output_dir), run_start, journal
        self.writer = AsyncStreamWriter()
        if saved is not None:
            from gpuwrf.runtime.restart_store import _hash_file
            for key, entries in saved["outputs"].items():
                if key not in {p[0].name for p in plans}:
                    raise ValueError("restart regional stream names differ")
                for entry in entries:
                    path = Path(entry["path"])
                    if path.parent.resolve() != self.output_dir.resolve() or not path.is_file() or _hash_file(path) != entry["sha256"]:
                        raise ValueError("restart regional frame missing, corrupt or outside output stream")
                self.writer.written[key] = [e["path"] for e in entries]
            if self.journal is None:
                raise ValueError("regional stream restart requires an authenticated journal")
            self.journal.recover(saved["outputs"])

    def flush(self, carries):
        """Transfer only bounded completed frames at an existing host boundary.

        Unfinished window reductions remain resident. Count placement matches the
        seed so a flush does not change the model JIT's committed-input signature.
        """
        import jax
        import jax.numpy as jnp
        result = dict(carries)
        per_domain = {name: list(c.regional_streams) for name,c in carries.items() if c.regional_streams is not None}
        for spec, index, points, lat, lon, ids, dt in self.plans:
            b = per_domain[spec.domain][index]
            count, steps, frames, invalid = jax.device_get((b.count, b.end_steps, b.frames, b.invalid))
            count = int(count)
            if count < 0 or count > b.frames.shape[0]:
                raise RuntimeError(f"{spec.name}: regional stream buffer overflow")
            if bool(invalid) or not np.isfinite(frames[:count]).all():
                raise ValueError(f"{spec.name}: regional stream sampled NaN/Inf")
            if count:
                self.writer.submit(spec.name, self.output_dir, spec, points, lat, lon, ids, dt,
                                   self.run_start, np.array(steps[:count], copy=True),
                                   np.array(frames[:count], copy=True), self.journal)
            zero = jax.device_put(jnp.int32(0), device=b.count.device)
            per_domain[spec.domain][index] = replace(b, count=zero)
        for name, buffers in per_domain.items():
            result[name] = carries[name].replace(regional_streams=tuple(buffers))
        return result

    def checkpoint_state(self):
        from gpuwrf.runtime.restart_store import output_receipts
        self.writer.drain()
        return {"history_stream": self.journal.stream if self.journal is not None else None,
                "outputs": output_receipts(self.writer.written)}

    def close(self):
        self.writer.close()

    def report(self):
        return {"files": self.writer.written, "max_root_steps": self.max_root_steps,
                "streams": [p[0].name for p in self.plans]}


def prepare_streams(config, names, carries, dt_by_domain, run_start, *, resumed=False,
                    restart_store=None, restart_identity=None, saved=None):
    path = Path(config.input_dir) / FILENAME
    existing = any(getattr(c, "regional_streams", None) is not None for c in carries.values())
    if not path.is_file():
        if existing or saved is not None:
            raise ValueError("regional stream configuration is missing at restart")
        return None, carries
    specs = read_streams(path)
    if not specs:
        if existing or saved is not None:
            raise ValueError("regional streams cannot be disabled at restart")
        return None, carries
    if resumed and (not existing or saved is None):
        raise ValueError("restart lacks regional stream state (E78)")
    from gpuwrf.io.netcdf_lock import Dataset
    import jax
    from .device import seed, SCHEMA_VERSION, BUFFER_CONTRACT
    plans, buffers, max_steps = [], {name: [] for name in names}, None
    for spec in specs:
        if spec.domain not in names:
            raise ValueError(f"regional stream {spec.name}: domain {spec.domain} is not loaded")
        carry = carries[spec.domain]
        ny, nx = carry.state.theta.shape[-2:]
        for field, _ in spec.channels:
            group, member, _ = FIELDS[field]
            value = getattr(getattr(carry, group, None), member, None)
            if value is None or value.shape != (ny,nx):
                raise ValueError(f"{spec.name}: {field} requires held {group}.{member} on the mass grid")
        dt, root_dt = float(dt_by_domain[spec.domain]), float(dt_by_domain[names[0]])
        interval = spec.steps(dt)
        with Dataset(Path(config.input_dir) / f"wrfinput_{spec.domain}") as ds:
            lat, lon = np.asarray(ds["XLAT"][0]), np.asarray(ds["XLONG"][0])
        y, x, ids = region_points(spec, lat, lon)
        if lat.shape != (ny,nx):
            raise ValueError("regional stream coordinate/state shape mismatch")
        capacity = max(spec.buffer_frames, math.ceil(root_dt/(dt*interval)))
        index = len(buffers[spec.domain])
        if resumed:
            restored = carry.regional_streams
            if restored is None or index >= len(restored):
                raise ValueError("restart regional stream plan missing (E78)")
            b = restored[index]
            if (b.schema_version != SCHEMA_VERSION or b.signature != BUFFER_CONTRACT or
                    b.channels != spec.channels or b.interval != interval or
                    b.frames.shape != (capacity,len(spec.channels),len(x)) or
                    not np.array_equal(b.x, x) or not np.array_equal(b.y, y) or int(b.count) != 0):
                raise ValueError("restart regional stream schema/region/configuration differs (E78)")
        else:
            b = jax.device_put(seed(carry, spec, y, x, interval, capacity), device=carry.state.theta.device)
        buffers[spec.domain].append(b)
        steps = max(1, int(math.floor(capacity*interval*dt/root_dt + 1.e-9)))
        max_steps = steps if max_steps is None else min(max_steps, steps)
        plans.append((spec, index, (y,x), lat[y,x], lon[y,x], ids, dt))
    if resumed and any(len(buffers[n]) != len(carries[n].regional_streams or ()) for n in names):
        raise ValueError("restart contains additional regional stream plans")
    result = {n: c.replace(regional_streams=tuple(buffers[n]) or None) for n,c in carries.items()}
    journal = None
    if restart_store is not None:
        from gpuwrf.runtime.restart_store import HistoryJournal
        identity = dict(restart_identity, output_dir=str((Path(config.output_dir)/"streams").resolve()))
        journal = HistoryJournal(restart_store, identity, saved["history_stream"] if saved is not None else None)
    runtime = StreamRuntime(plans, max_steps, Path(config.output_dir)/"streams", run_start, journal, saved)
    return runtime, result
