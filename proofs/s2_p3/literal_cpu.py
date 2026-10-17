"""No-FMA REAL4 execution of the port using the Fortran host libm intrinsics.

Validation only: array arithmetic rounds at every NumPy REAL operation, with
libm scalar intrinsics at WRF's REAL/DOUBLE precision. The original module is
the independent reference. This executes the port's control flow and formulas;
it never invokes the original P3 routine or changes its fixtures.
"""
import argparse
import ctypes
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

from comparison import ComparisonError, checked_inventory, compare_field


libm = ctypes.CDLL('libm.so.6')


def intrinsic(name, *values):
    dtype = np.result_type(*(np.asarray(v).dtype for v in values))
    real = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    fn = getattr(libm, name + ('' if real == ctypes.c_double else 'f'))
    fn.argtypes = [real] * len(values)
    fn.restype = real
    arrays = np.broadcast_arrays(*map(np.asarray, values))
    result = np.fromiter((fn(*x) for x in zip(*(a.flat for a in arrays))), dtype=dtype, count=arrays[0].size)
    return result.reshape(arrays[0].shape).view(Real)


class Real(np.ndarray):
    def __getitem__(self, key):
        # The port's masked SIMD lanes still issue gathers. Match JAX's
        # ordinary indexing (negative wrap then clamp) only in this executor;
        # no scientific value or product-side cleanup is changed here.
        if isinstance(key, tuple):
            fixed = []
            for axis, index in enumerate(key):
                if isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer):
                    index = np.where(index < 0, index + self.shape[axis], index)
                    index = np.clip(index, 0, self.shape[axis] - 1)
                fixed.append(index)
            key = tuple(fixed)
        return super().__getitem__(key)

    def __pow__(self, exponent):
        e = np.asarray(exponent)
        if e.ndim == 0 and np.isfinite(e) and int(e) == e and int(e) >= 0:
            n = int(e)
            # gfortran's small integer powers, including x**6 = (x*x*x)**2.
            if n == 0:
                return np.ones_like(self)
            if n == 1:
                return self
            if n in (2, 3):
                return self * self if n == 2 else (self * self) * self
            if n in (4, 6):
                v = self ** (n // 2)
                return v * v
            if n == 5:
                return (self ** 4) * self
        if e.ndim == 0 and e == .5:
            return intrinsic('sqrt', self)
        return intrinsic('pow', self, np.asarray(exponent, dtype=self.dtype))


class LiteralNumpy:
    def __getattr__(self, name):
        if name in ('exp', 'log', 'log10', 'sqrt'):
            return lambda x: intrinsic(name, x)
        fn = getattr(np, name)
        if not callable(fn) or isinstance(fn, type):
            return fn
        def wrap(value):
            if isinstance(value, tuple):
                return tuple(map(wrap, value))
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                return value.view(Real)
            return value
        return lambda *a, **kw: wrap(fn(*a, **kw))

    def asarray(self, value, dtype=None):
        a = np.asarray(value, dtype=dtype)
        return a.view(Real) if np.issubdtype(a.dtype, np.floating) else a


def no_fma(op, *values):
    if op == 'identity':
        return values[0]
    a, b = values
    return {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide}[op](a, b)


def while_loop(condition, body, carry):
    for _ in range(10000):
        if not condition(carry):
            return carry
        carry = body(carry)
    raise RuntimeError('literal sedimentation exceeded source loop budget')


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--schemes', type=int, nargs='+', choices=(50,51,52,53), default=(50,51,52,53))
    p.add_argument('--fixtures', type=Path, default=Path(__file__).with_name('fixtures'))
    args = p.parse_args(argv)
    assert jax.devices()[0].platform == 'cpu'
    modules = [importlib.import_module('gpuwrf.physics.p3.' + name) for name in
               ('column','lookup','properties','process_support','initial_generated','processes_generated','final_generated')]
    facade = LiteralNumpy()
    for mod in modules:
        mod.jnp = facade
        if hasattr(mod, '_rounded_real'):
            mod._rounded_real = no_fma
        if hasattr(mod, 'gamma'):
            mod.gamma = lambda x: intrinsic('tgamma', x)
    column = modules[0]
    column.qzi_decode = lambda q,n: facade.where(n>=np.float32(1.e-14),q**2/facade.where(n>=np.float32(1.e-14),n,np.float32(1)),np.float32(0))
    column.qzi_encode = lambda z,n: facade.where((n>=np.float32(1.e-14))&(z>=np.float32(1.e-30)),intrinsic('sqrt',z*n),np.float32(0))
    report = {'mode': 'literal REAL4 no-FMA host libm', 'schemes': {}, 'BYTE': True, 'frozen_failures': 0}
    report.update(validation_failures=0, nonfinite_failures=0, byte_failures=0)
    try:
        with patch.object(jax.lax, 'while_loop', while_loop), np.errstate(all='ignore'):
            for mp in args.schemes:
                with np.load(args.fixtures / f'columns_mp{mp}.npz', allow_pickle=False) as f:
                    data = {k: f[k] for k in f.files}
                checked_inventory(data, f'fixture.mp{mp}')
                fields = {k: facade.asarray(v[...,0]) for k,v in data.items() if v.ndim == 3}
                if mp != 52:
                    for name in ('qi2','ni2','qir2','bir2','vm2','di2','rhoi2'):
                        fields.pop(name, None)
                if mp != 53:
                    fields.pop('qzi', None)
                tables = column.selected_tables(mp)
                records = {}
                for it in range(1,4):
                    out = column._step_impl(fields, np.float32(18), it, tables, mp=mp)
                    checked_inventory(out, f'candidate.mp{mp}.step{it}', mp=mp, reference=data)
                    scores = {}
                    for name, value in out.items():
                        truth = data[name][...,it]
                        score = compare_field(name, value, truth, literal=True)
                        scores[name] = {key: score[key] for key in ('different','bad','max_abs','finite')}
                        report['BYTE'] &= score['BYTE']
                        report['frozen_failures'] += score['bad']
                        report['byte_failures'] += score['different']
                    records[str(it)] = scores
                    print(mp,it,{k:v for k,v in scores.items() if v['bad']}, flush=True)
                    fields = {k: out.get(k,v) for k,v in fields.items()}
                report['schemes'][str(mp)] = records
                args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except ComparisonError as error:
        report['BYTE'] = False
        report['validation_failures'] += error.count
        report['nonfinite_failures'] += error.nonfinite
        report['validation_error'] = str(error)
    report['failure_count'] = (max(report['frozen_failures'], report['byte_failures']) +
                               report['validation_failures'])
    report['verdict'] = 'PASS' if report['BYTE'] and report['failure_count'] == 0 else 'FAIL'
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('BYTE',report['BYTE'],'frozen_failures',report['frozen_failures'],flush=True)
    print('failure_count', report['failure_count'], report['verdict'], flush=True)
    assert report['BYTE'] and report['failure_count'] == 0, 'literal P3 differs from independently compiled original WRF'


if __name__ == '__main__':
    main()
