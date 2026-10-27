"""SDS key determinism and E45 separation; no forecast/fidelity comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck


class EmptyStatic:
    def tree_flatten(self):
        return (), ()


def captured(spec):
    @jax.jit
    def body():
        return jnp.zeros(spec.shape, spec.dtype) + (2 if spec.weak_type else 1) + (4 if spec.is_ref else 0)
    return body


def test_equal_descriptors_never_materialize_object_pointer_bytes(monkeypatch):
    original = np.asarray
    seen = []

    def checked(obj, *args, **kwargs):
        if isinstance(obj, jax.ShapeDtypeStruct):
            seen.append(obj)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(np, 'asarray', checked)
    a, b = (jax.ShapeDtypeStruct((2, 3), np.float32) for _ in range(2))
    assert a is not b
    assert ck.canonical_digest(a) == ck.canonical_digest(b)
    assert seen == []
    assert ck.canonical_digest(a) != ck.canonical_digest(np.zeros((2, 3), np.float32))


def test_descriptor_metadata_separates_programs():
    specs = [jax.ShapeDtypeStruct((), np.float32),
             jax.ShapeDtypeStruct((), np.float32, weak_type=True),
             jax.ShapeDtypeStruct((), np.float32, is_ref=True),
             jax.ShapeDtypeStruct((2,), np.float32),
             jax.ShapeDtypeStruct((2,), np.float64)]
    groups = {}
    records = []
    for spec in specs:
        fn = captured(spec)
        key = ck.program_key(fn, (), {}, EmptyStatic())
        hlo = hashlib.sha256(fn.lower().as_text().encode()).hexdigest()
        groups.setdefault(key, set()).add(hlo)
        records.append((key, hlo))
    assert all(len(hlo) == 1 for hlo in groups.values())
    assert len(groups) == len(specs)
    assert records[0][1] != records[1][1]  # Astra's weak_type counterexample.
    assert records[0][1] != records[2][1]
    assert ck.program_key(captured(jax.ShapeDtypeStruct((), np.float32)), (), {}, EmptyStatic()) == records[0][0]


@jax.tree_util.register_pytree_node_class
class StaticDescriptor:
    def __init__(self, spec):
        self.spec = spec

    def tree_flatten(self):
        return (), self.spec

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)


def unsupported_specs():
    from jax._src.core import ManualAxisType
    from jax._src.layout import Layout
    from jax.experimental.layout import Format
    sharding = jax.sharding.SingleDeviceSharding(jax.devices('cpu')[0])
    return [jax.ShapeDtypeStruct((2, 3), np.float32, sharding=sharding),
            jax.ShapeDtypeStruct((2, 3), np.float32, manual_axis_type=ManualAxisType(varying=frozenset({'x'}))),
            jax.ShapeDtypeStruct((2, 3), np.float32, sharding=Format(Layout((1, 0)), sharding))]


def test_unsupported_static_metadata_refuses_full_key_even_in_pytree_aux():
    for spec in unsupported_specs():
        with pytest.raises(ck._UnsupportedAbstractMetadata):
            ck.canonical_digest(spec)
        assert ck.cheap_key(captured(spec), (), {}, EmptyStatic()) is None
        obj = StaticDescriptor(spec)
        with pytest.raises(ck._UnsupportedAbstractMetadata):
            ck._structure_fingerprint((obj,))
        assert ck.cheap_key(lambda x: x, (obj,), {}, EmptyStatic()) is None


def test_unsupported_code_constant_cannot_drop_code_identity():
    # CodeType is explicitly part of the canonical walk. Refusing an abstract
    # constant must not be caught by fn_identity_hash's ordinary best-effort path.
    def template():
        return (1).shape[0]
    spec = unsupported_specs()[0]
    consts = tuple(spec if type(value) is int and value == 1 else value
                   for value in template.__code__.co_consts)
    fn = types.FunctionType(template.__code__.replace(co_consts=consts), globals(), 'constant_descriptor')
    with pytest.raises(ck._UnsupportedAbstractMetadata):
        ck.fn_identity_hash(fn)
    assert ck.cheap_key(fn, (), {}, EmptyStatic()) is None


def test_concrete_encodings_remain_v043_byte_identical():
    # Baseline: immutable v0.4.3 bundle pin 888804310b; no SDS in these values.
    fixtures = [
        (np.arange(6, dtype=np.float32).reshape(2, 3), '5367ea41da7c81bfae899bf75335f232a87c36e3216cc561e6601b6df406b684'),
        (np.array([-0., 1., 3.25], np.float64), 'ecb53fa7a826e3a8edd0863855ac14811d65d92ca87c631b09a3d9345d05bc5f'),
        (np.array([1, -2, 3], np.int32), '1a058d82c5d3b4ee6e31896b357239bf300bbd281c3abbca0ef65838ee996140'),
        ({'flag': True, 'value': np.array([2., 3.], np.float32), 'meta': (None, 'dt', 54.)},
         '8bc84ff9b4d45f75d9d194988f7eab3f76c0c2a547d6ecd867671ef3eab933b1'),
    ]
    for value, expected in fixtures:
        assert ck.canonical_digest(value) == expected
    assert ck.canonical_digest(np.zeros(2, np.float32)) != ck.canonical_digest(np.ones(2, np.float32))


def test_equal_descriptors_are_fresh_process_stable():
    src = str(Path(__file__).resolve().parents[1] / 'src')
    code = """import json,jax,numpy as np
from gpuwrf.runtime.aot_cheap_key import canonical_digest
print(json.dumps([canonical_digest(jax.ShapeDtypeStruct((2,3),np.float32)) for _ in range(2)]))
"""
    records = []
    for seed in ('1', '73'):
        env = {**os.environ, 'PYTHONPATH': src, 'PYTHONHASHSEED': seed,
               'JAX_PLATFORMS': 'cpu', 'GPUWRF_JAX_CACHE': '0', 'CUDA_VISIBLE_DEVICES': ''}
        proc = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        records += json.loads(proc.stdout.splitlines()[-1])
    assert len(set(records)) == 1
