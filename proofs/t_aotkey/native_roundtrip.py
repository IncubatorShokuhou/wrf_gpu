"""Exact-input GPU diagnostic; no forecast/fidelity or speed claim.

Run only in a manager window under with_gpu_lock, on a detached source tree.
The E1 reproduction should use original commit 86e63037 and the Z table root.
Original compiled execution before/after export and AOT reload consume separate
copies of the SAME inputs. Retained artifacts are never written or weakened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--retained', type=Path)
    args = parser.parse_args()
    if os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1':
        parser.error('official GPU lock required before importing JAX')
    source = args.source.resolve()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if subprocess.run(['git','symbolic-ref','-q','HEAD'],cwd=source,capture_output=True).returncode == 0:
        parser.error('a detached worktree is required')
    if commit != args.commit or subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=source,text=True).strip():
        parser.error('source must be the declared clean immutable commit')
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out/'cache'
    os.environ.update(GPUWRF_WRF_ROOT=str(args.tables.resolve()),
                      GPUWRF_JAX_CACHE_DIR=str(cache), JAX_COMPILATION_CACHE_DIR=str(cache),
                      GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(args.out/'autotune'),
                      CUDA_CACHE_PATH=str(args.out/'cuda'), TRITON_CACHE_DIR=str(args.out/'triton'),
                      GPUWRF_LAYOUT_PIN='ac_carry_cols_cum', GPUWRF_JAX_CACHE='1',
                      GPUWRF_NESTED_AOT='1', XLA_PYTHON_CLIENT_PREALLOCATE='true',
                      XLA_PYTHON_CLIENT_MEM_FRACTION='.25')
    sys.path.insert(0,str(source/'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime import aot_executable as ax, aot_precompile as ap, operational_mode as op
    assert jax.devices()[0].platform == 'gpu', jax.devices()
    assert Path(op.__file__).resolve().is_relative_to(source/'src')
    report = dict(commit=commit, platform='gpu', source=str(source), case=str(args.case),
                  tables=str(args.tables), cache=str(cache), comparisons=[],
                  classification='cache/execution regression; not CPU-WRF fidelity evidence')
    report['inputs'] = {}
    for name in ('namelist.input','wrfinput_d01','wrfbdy_d01','wrflowinp_d01'):
        path = args.case/name
        if path.is_file():
            with path.open('rb') as stream:
                report['inputs'][name] = hashlib.file_digest(stream,'sha256').hexdigest()
    def save():
        (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    def copied(tree):
        result = jax.tree.map(lambda x: x.copy() if isinstance(x,jax.Array) else x, tree)
        return jax.block_until_ready(result)
    def compare(label, left, right):
        a, at = jax.tree_util.tree_flatten(left)
        b, bt = jax.tree_util.tree_flatten(right)
        row = dict(label=label, leaves=len(a), same_tree=at==bt, different=[],nonfinite=[])
        assert at==bt and len(a)==len(b), row
        for i,(x,y) in enumerate(zip(a,b,strict=True)):
            x,y = np.asarray(x),np.asarray(y)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                row['nonfinite'].append(i)
            if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():
                row['different'].append(dict(leaf=i, shape=x.shape,
                                             max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float))))))
        report['comparisons'].append(row)
        save()
        return not row['different'] and not row['nonfinite']
    def roundtrip(name, fn, inputs, static=None, *, retained=None):
        call_args, kwargs = inputs
        static = static or {}
        lowered = fn.lower(*call_args, **kwargs, **static)
        hlo = ax.hlo_sha256_from_lowered(lowered)
        started = time.perf_counter()
        compiled = lowered.compile(ax.export_compile_options() if name=='d01' else None)
        before = jax.block_until_ready(compiled(*copied(call_args), **copied(kwargs)))
        status = ap._serialize_domain_blob(name, compiled, cache, hlo_sha256=hlo, lowered=lowered)
        report.setdefault('programs',[]).append(dict(name=name,hlo=hlo,capture=status,
                                                   compile_execute_export_s=time.perf_counter()-started))
        save()
        assert status['aot_written'], status
        blob_path, meta_path = ap._aot_blob_paths(name,cache,hlo_sha256=hlo)
        with meta_path.open('rb') as stream:
            meta = pickle.load(stream)
        loaded = ax.load(blob_path.read_bytes(),meta)
        after = jax.block_until_ready(compiled(*copied(call_args), **copied(kwargs)))
        restored = jax.block_until_ready(loaded(*copied(call_args), **copied(kwargs)))
        compare(name+':before-vs-after-export', before, after)
        compare(name+':original-vs-reload', before, restored)
        if retained:
            with retained.with_suffix('.meta').open('rb') as stream:
                original_meta = pickle.load(stream)
            if original_meta.hlo_sha256 == hlo:
                try:
                    original = ax.load(retained.read_bytes(),original_meta)
                    compare(name+':retained-vs-fresh',before,
                            jax.block_until_ready(original(*copied(call_args), **copied(kwargs))))
                except Exception as exc:
                    report['retained_refusal'] = dict(reason='loader/ABI refused; guards retained',
                                                       error=f'{type(exc).__name__}: {exc}')
                    save()
            else:
                report['retained_refusal'] = dict(reason='different exact lowered HLO',live=hlo,
                                                   retained=original_meta.hlo_sha256)
                save()
        return before
    def init_call(fn, *inputs, **kwargs):
        static = {k:v for k,v in kwargs.items() if k in {'debug','with_clear_sky','column_tile_cols'}}
        dynamic = {k:v for k,v in kwargs.items() if k not in static}
        return roundtrip('init_'+fn.__name__,fn,(inputs,dynamic),static)
    config = NestedPipelineConfig(input_dir=args.case, output_dir=args.out/'unused-history',
                                  proof_dir=args.out/'proofs', hours=3,max_dom=1)
    _,bundles,_,_,_,carries = _load_domains(config,('d01',),_init_kernel_call=init_call)
    nl = bundles['d01'].namelist
    report['work'] = {name:getattr(nl,name) for name in
                      ('dt_s','acoustic_substeps','mp_physics','bl_pbl_physics',
                       'sf_sfclay_physics','sf_surface_physics','cu_physics',
                       'ra_sw_physics','ra_lw_physics','radiation_cadence_steps')}
    clock = op.build_clock_base(nl)
    roundtrip('d01',op._advance_chunk_fori,
              ((carries['d01'],nl,jnp.asarray(1,jnp.int32),clock),
               dict(n_steps=1,cadence=int(nl.radiation_cadence_steps))),retained=args.retained)
    report['bitwise'] = all(row['same_tree'] and not row['different'] and not row['nonfinite']
                           for row in report['comparisons'])
    report['e41'] = 'PENDING scripts/wave_cubin_gate.py on declared private cache before cleanup'
    save()
    return 0 if report['bitwise'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
