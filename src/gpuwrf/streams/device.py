"""Compact REAL32 window reducers and bounded frames; no host step-loop work."""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .config import FIELDS

SCHEMA_VERSION = 1
BUFFER_CONTRACT = "regional-window-v1"

def validate_schema(buffers, version):
    """E78: never replace a missing/malformed active observer by fresh zeros."""
    if buffers is None:
        return
    import numpy as np
    if version != SCHEMA_VERSION or not isinstance(buffers, tuple) or not buffers:
        raise ValueError("restart regional stream schema missing or unsupported (E78)")
    for b in buffers:
        if not isinstance(b, StreamBuffer) or b.schema_version != SCHEMA_VERSION:
            raise ValueError("restart regional stream buffer schema differs (E78)")
        n = len(b.channels)
        if (not n or b.interval < 1 or b.x.ndim != 1 or b.x.shape != b.y.shape or
                b.running.shape != (n,len(b.x)) or b.baseline.shape != b.running.shape or
                b.frames.ndim != 3 or b.frames.shape[1:] != b.running.shape or
                b.end_steps.shape != b.frames.shape[:1]):
            raise ValueError("restart regional stream array shapes differ (E78)")
        for leaf in (b.running,b.baseline,b.frames):
            if np.dtype(leaf.dtype) != np.dtype("float32"):
                raise ValueError("restart regional stream requires REAL32 arrays (E78)")
        for leaf in (b.x,b.y,b.end_steps,b.count,b.samples,b.invalid):
            if np.dtype(leaf.dtype) != np.dtype("int32"):
                raise ValueError("restart regional stream requires int32 clocks/indices (E78)")
        if (b.count.shape != () or b.samples.shape != () or b.invalid.shape != () or
                not 0 <= int(b.count) <= b.frames.shape[0] or not 0 <= int(b.samples) < b.interval):
            raise ValueError("restart regional stream counters differ (E78)")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StreamBuffer:
    x: object
    y: object
    running: object
    baseline: object
    frames: object
    end_steps: object
    count: object
    samples: object
    invalid: object
    channels: tuple
    interval: int
    signature: str
    schema_version: int = SCHEMA_VERSION

    def tree_flatten(self):
        return ((self.x, self.y, self.running, self.baseline, self.frames, self.end_steps,
                 self.count, self.samples, self.invalid),
                (self.channels, self.interval, self.signature, self.schema_version))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def sample(carry, channels, y, x):
    gathered = {}
    for name, _ in channels:
        if name not in gathered:
            group, member, _ = FIELDS[name]
            source = getattr(getattr(carry, group), member)
            # Gather first: no domain-wide diagnostic recomputation or temporaries.
            gathered[name] = jnp.asarray(source[y, x], jnp.float32)
    return jnp.stack([gathered[name] for name, _ in channels])


def seed(carry, spec, y, x, interval, capacity):
    y, x = jnp.asarray(y, jnp.int32), jnp.asarray(x, jnp.int32)
    current = sample(carry, spec.channels, y, x)
    return StreamBuffer(x, y, jnp.zeros_like(current), current,
                        jnp.zeros((capacity, *current.shape), jnp.float32),
                        jnp.zeros((capacity,), jnp.int32), jnp.int32(0), jnp.int32(0),
                        jnp.int32(0), spec.channels, interval, BUFFER_CONTRACT)


def record_buffer(buf, values, step_index):
    """Samples (window_start, window_end]; precipitation = endpoint difference.

    Mean is the arithmetic mean of uniform own-step samples, no t0 sample. Every
    sampled NaN/Inf poisons its window, including interior instant-only samples.
    """
    rows = []
    for k, (_, statistic) in enumerate(buf.channels):
        value, previous = values[k], buf.running[k]
        if statistic == "mean":
            value = previous + value
        elif statistic == "min":
            value = jnp.where(buf.samples == 0, value, jnp.minimum(previous, value))
        elif statistic == "max":
            value = jnp.where(buf.samples == 0, value, jnp.maximum(previous, value))
        elif statistic == "accumulation":
            value = value - buf.baseline[k]
        rows.append(value)
    running = jnp.stack(rows)
    samples = buf.samples + jnp.int32(1)
    invalid = buf.invalid | (~jnp.all(jnp.isfinite(values))).astype(jnp.int32)
    updated = replace(buf, running=running, samples=samples, invalid=invalid)

    def finish(b):
        output = jnp.stack([b.running[k] / b.samples.astype(jnp.float32) if stat == "mean"
                            else b.running[k] for k, (_, stat) in enumerate(b.channels)])
        output = jnp.where(b.invalid != 0, jnp.float32(jnp.nan), output)
        return replace(b, frames=b.frames.at[b.count].set(output),
                       end_steps=b.end_steps.at[b.count].set(jnp.asarray(step_index, jnp.int32)),
                       count=b.count+jnp.int32(1), samples=jnp.int32(0), invalid=jnp.int32(0),
                       running=jnp.zeros_like(b.running), baseline=values)

    return jax.lax.cond(step_index % buf.interval == 0, finish, lambda b: b, updated)


def record_step(carry, step_index):
    buffers = tuple(record_buffer(b, sample(carry, b.channels, b.y, b.x), step_index)
                    for b in carry.regional_streams)
    return carry.replace(regional_streams=buffers)
