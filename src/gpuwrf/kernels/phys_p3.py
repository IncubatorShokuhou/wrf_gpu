"""P3 per-column Pallas entry: local adaptive loops and gathered table refs.

CPU lowering/interpretation is available for qualification without a GPU.
GPU execution and resource/performance qualification are explicitly pending.
"""
import math

import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.physics.p3.column import _step_impl, selected_tables
from gpuwrf.physics.p3.constants import CONSTANTS, DERF_A, DERF_B
from gpuwrf.physics.p3.process_support import NATIVE_REAL, GAMMA_FACTORIALS, DERF_TABLES, MP53_PREDICATE_RESHAPE


class TableRef:
    _p3_table_ref = True

    def __init__(self, ref, load=plt.load):
        self.ref, self.shape, self.ndim = ref, ref.shape, len(ref.shape)
        self.load = load

    def __getitem__(self, index):
        return self.load(self.ref.at[index])


def full_column(fields, dt, timestep, *, mp, interpret=False, output_names=None):
    if mp in (52, 53):
        from gpuwrf.kernels._p3_memory import load, store
    else:
        load, store = plt.load, plt.store
    shape = fields['qv'].shape
    nz, ncol = shape[-1], math.prod(shape[:-1])
    width = 1 << (nz-1).bit_length()
    names = tuple(fields)
    arrays = tuple(jnp.asarray(fields[name], jnp.float32).reshape((ncol, nz)) for name in names)
    tables = selected_tables(mp)
    tables['gamma_factorials'] = jnp.asarray(tuple(float(math.factorial(k)) for k in range(35)), jnp.float32)
    if mp != 50:
        tables.update(derf_a=jnp.asarray(DERF_A,jnp.float32), derf_b=jnp.asarray(DERF_B,jnp.float32))
    table_names = tuple(tables)
    table_arrays = tuple(tables.values())
    constant_names = tuple(CONSTANTS)
    constant_array = jnp.asarray(tuple(CONSTANTS.values()), jnp.float32)
    # Trace shapes only; this never executes a separate reference column.
    with jax.enable_x64(True):
        example = {name: jax.ShapeDtypeStruct((nz,), jnp.float32) for name in names}
        output = jax.eval_shape(lambda x, d, i: _step_impl(x, d, i, tables, mp=mp), example,
                                jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct((), jnp.int32))
    output_names = tuple(output) if output_names is None else tuple(output_names)
    if len(set(output_names)) != len(output_names) or any(name not in output for name in output_names):
        raise ValueError('P3 requested duplicate/unknown column output')
    output_shapes = tuple(jax.ShapeDtypeStruct((ncol, nz) if output[name].ndim else (ncol,), jnp.float32)
                          for name in output_names)

    def kernel(*refs):
        col = pl.program_id(0)
        level = jnp.arange(width, dtype=jnp.int32)
        valid = level < nz
        pads = {'th': 300., 'th_old': 300., 'p': 100000., 'exner': 1., 'dz': 250.}
        local = {name: load(ref.at[col, level], mask=valid, other=pads.get(name, 0.))
                 for name, ref in zip(names, refs[:len(names)])}
        start = len(names)
        local_tables = {name: TableRef(ref, load) for name, ref in
                        zip(table_names, refs[start:start+len(table_names)])}
        token = NATIVE_REAL.set(not interpret)
        predicate_token = MP53_PREDICATE_RESHAPE.set(mp == 53)
        gamma_token = GAMMA_FACTORIALS.set(local_tables['gamma_factorials'])
        derf_token = DERF_TABLES.set((local_tables['derf_a'],local_tables['derf_b']) if mp!=50 else None)
        try:
            offset = start+len(table_names)
            local_constants = {name: plt.load(refs[offset].at[index]) for index,name in enumerate(constant_names)}
            local_dt, local_it = plt.load(refs[offset+1].at[()]), plt.load(refs[offset+2].at[()])
            result = _step_impl(local, local_dt, local_it, local_tables, mp=mp,
                                physical_constants=local_constants)
        finally:
            NATIVE_REAL.reset(token)
            MP53_PREDICATE_RESHAPE.reset(predicate_token)
            GAMMA_FACTORIALS.reset(gamma_token)
            DERF_TABLES.reset(derf_token)
        for name, ref in zip(output_names, refs[start+len(table_names)+3:]):
            if output[name].ndim:
                store(ref.at[col, level], result[name], mask=valid)
            else:
                store(ref.at[col], result[name])

    with jax.enable_x64(True):
        result = pl.pallas_call(kernel, grid=(ncol,), out_shape=output_shapes,
                                interpret=interpret,
                                compiler_params=plt.CompilerParams(num_warps=4 if mp in (52, 53) else 2),
                                name=f'p3_mp{mp}_column')(*arrays, *table_arrays, constant_array,
                                                        jnp.asarray(dt, jnp.float32), jnp.asarray(timestep, jnp.int32))
    return {name: value.reshape(shape if output[name].ndim else shape[:-1])
            for name, value in zip(output_names, result)}
