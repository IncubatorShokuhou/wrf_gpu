"""Trace both measured domains/initial and ordinary carry shapes for source reuse.

Consumes a sealed CPU initialization seed; executes no forecast timesteps. The
CPU shims and resolved environment are exactly those of coupled_gate.py.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

root, seed, output = map(Path, sys.argv[1:4])
seal = json.loads(seed.with_suffix('.json').read_text())
assert os.environ['JAX_PLATFORMS'] == 'cpu'
os.environ.update(seal['signature']['env'])
sys.path[:0] = [str(root / 'src'), str(root / 'proofs/o1_smag3d')]
import cpu_safe
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as pt
cpu_safe.install()
assert jax.devices()[0].platform == 'cpu' and jax.config.x64_enabled
real_pallas_call = pl.pallas_call
# Reuse, rather than independently reimplement, the measured CPU arithmetic.
tree = ast.parse((root / 'proofs/u2_twoway/coupled_gate.py').read_text())
definitions = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in ('interpret_call', 'cpu_inline_asm')]
assert len(definitions) == 2
exec(compile(ast.Module(body=definitions, type_ignores=[]), '<coupled-CPU-shims>', 'exec'))
pl.pallas_call = interpret_call
pt.elementwise_inline_asm = cpu_inline_asm
from gpuwrf.contracts import state as state_contract
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.domain_tree import DomainTree
state_contract._gpu_device = lambda: jax.devices('cpu')[0]
seed_bytes = seed.read_bytes()
assert hashlib.sha256(seed_bytes).hexdigest() == seal['sha256']
hierarchy, bundles, meta, run_start, dts, carries = pickle.loads(seed_bytes)
del seed_bytes
domain_tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=True,
    feedback_smooth_option=2, feedback_spec_zone=1)

def constant_hashes(value):
    if hasattr(value, 'jaxpr') and hasattr(value, 'consts'):
        for item in value.consts:
            array = np.asarray(item)
            yield dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())
        yield from constant_hashes(value.jaxpr)
    elif hasattr(value, 'eqns'):
        for eqn in value.eqns:
            yield from constant_hashes(eqn.params)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from constant_hashes(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from constant_hashes(item)

report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    seed_sha256=seal['sha256'], backend='cpu', x64_enabled=bool(jax.config.x64_enabled),
    resolved_env=seal['signature']['env'], cache='NONE', steps_executed=0, programs={})
start = time.monotonic()
for name in ('d01', 'd02'):
    ns = domain_tree.domains[name].namelist
    om._resolve_operational_suite(ns)
    cb = om.build_clock_base(ns)
    c = carries[name]
    for variant in ('initial', 'ordinary'):
        jax.clear_caches()
        fn = lambda carry: om._advance_chunk_fori(carry, ns, jnp.int32(1), cb,
            n_steps=1, cadence=int(ns.radiation_cadence_steps))
        jx = jax.make_jaxpr(fn)(c)
        text = re.sub(r'name_and_src_info=[^\]\n]*', '', str(jx))
        text = re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?', '', text)
        report['programs'][name + '_' + variant] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
            chars=len(text), constants=list(constant_hashes(jx)),
            input_shapes=[dict(shape=list(x.shape), dtype=str(x.dtype)) for x in jax.tree.leaves(c)])
        c = jax.eval_shape(fn, c)
        report['wall_s'] = time.monotonic() - start
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(name, variant, report['programs'][name + '_' + variant]['sha256'], flush=True)
        del text, jx
