"""Warm both real nested qWT programs in one paid cache, then compare 3 h.

Control harness only: imports the immutable 3fa model snapshot. Priming uses the
production nested factory and one actual root cascade (counts 1/3/9), then exits
through the pipeline's writer-drain finally block. The same cache/flags supply
both later 3 h CLI runs. No new production physics or eager identity claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODEL = Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget/snapshot_p09')
PAID = Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget/NIGHT_1792792718433431644_987496')
CASE = Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227')
PIN = '3fa48d159766101770a787fb9947c0622c7c4456'


class PrimeComplete(BaseException):
    """Stop after the complete real cascade; pipeline finally drains writers."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def prereq():
    old = json.loads((PAID / 'probe/result.json').read_text())
    assert old['source_commit'] == PIN and old['platform'] == 'gpu'
    assert old['verdict'] == 'FIRST_STEP_BYTE' and old['byte_fields'] == old['total_fields'] == 183
    assert subprocess.check_output(['git', '-C', str(MODEL), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(MODEL), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    for name, record in old['input_files'].items():
        assert sha(CASE / name) == record['sha256'], name
    return old


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    old = prereq()
    if args.dry_run:
        print(json.dumps(dict(source=PIN, input_files=len(old['input_files']),
            cache=str(PAID/'cache/jax'), stages=['prime_off', 'prime_on', 'replay_off', 'replay_on'],
            root_steps_prime=1, root_steps_replay=200, gpu_started=False)))
        return
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert Path(os.environ['GPUWRF_JAX_CACHE_DIR']).resolve() == (PAID/'cache/jax').resolve()
    assert os.environ.get('GPUWRF_AOT_KEEP_TRACE_METADATA') == '1'
    sys.path.insert(0, str(MODEL / 'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf import cli
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import domain_tree as D
    assert jax.devices()[0].platform == 'gpu'
    assert Path(pipeline.__file__).resolve().is_relative_to(MODEL)
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / 'result.json'
    record = dict(schema='wrf_gpu2.w44_budget.night_probe.v2', source_commit=PIN,
        platform='gpu', scope='ON/OFF regression only, not original WRF fidelity',
        carried_first_step=dict(path=str(PAID/'probe/result.json'), sha256=sha(PAID/'probe/result.json'),
            byte_fields=183, total_fields=183),
        cache=str(PAID/'cache/jax'), rich_aot_metadata=True, input_files=old['input_files'],
        phases={}, verdict='FIRST_STEP_BYTE_CARRIED_FORWARD')
    atomic(output, record)
    original_load, original_segment = pipeline._load_domains, pipeline.run_operational_domain_tree
    current, final = {}, {}
    prime_arrays = {}

    def loaded(*a, **kw):
        values = original_load(*a, **kw)
        if current['on']:
            values = (*values[:-1], {name: c.replace(state=c.state.replace(
                mynn_qwt=jnp.zeros_like(c.state.qv, dtype=jnp.float32))) for name,c in values[-1].items()})
        return values

    def segment(*a, **kw):
        if current['prime']:
            kw = dict(kw, root_steps=1)
        begin = time.perf_counter()
        result = original_segment(*a, **kw)
        jax.block_until_ready(result.carries)
        current['segments'].append(dict(root_steps=int(kw['root_steps']), wall_s=time.perf_counter()-begin,
                                        own_steps=dict(result.own_steps)))
        final.clear(); final.update(result.carries)
        if current['prime']:
            inventory = D.nested_aot_report()
            atomic(args.out/'compiled_program_inventory.json', dict(metadata=dict(nested_aot=inventory)))
            current['aot'] = inventory
            raise PrimeComplete()
        return result

    def digest(carries, keep=False):
        rows, arrays = {}, {}
        for name, carry in carries.items():
            if current['on']:
                qwt = np.asarray(carry.state.mynn_qwt)
                assert qwt.shape == carry.state.qv.shape and np.isfinite(qwt).all() and np.count_nonzero(qwt)
            carry = carry.replace(state=carry.state.replace(mynn_qwt=None))
            for path, leaf in jax.tree.flatten_with_path(carry)[0]:
                value = np.asarray(leaf)
                assert np.isfinite(value).all(), f'non-finite {name}/{path}'
                key = f'{name}/{path}'
                rows[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                 sha256=hashlib.sha256(value.tobytes()).hexdigest())
                if keep:
                    arrays[key] = value.copy()
        return rows, arrays

    pipeline._load_domains, pipeline.run_operational_domain_tree = loaded, segment
    reference_rows = reference_files = None
    try:
        for label, on, prime in [('prime_off',False,True), ('prime_on',True,True),
                                 ('replay_off',False,False), ('replay_on',True,False)]:
            current.clear(); current.update(on=on, prime=prime, segments=[])
            final.clear()
            out = args.out/label
            start = time.perf_counter()
            try:
                rc = cli.main(['run','--input-dir',str(CASE),'--output-dir',str(out/'wrfout'),
                    '--scratch-dir',str(out/'scratch'),'--hours','3','--max-dom','3'])
                assert not prime, 'priming must stop after one complete cascade'
                assert rc == 0, (label,rc)
            except PrimeComplete:
                assert prime, 'unexpected early forecast stop'
            rows, arrays = digest(final, keep=prime)
            files = {str(f.relative_to(out/'wrfout')): sha(f) for f in sorted((out/'wrfout').glob('wrfout_d*')) if f.is_file()}
            arm = dict(wall_s=time.perf_counter()-start, segments=current['segments'],
                       terminal_leaf_sha256=rows, wrfout_sha256=files, files=len(files), prime_only=prime)
            if label == 'prime_off':
                prime_rows, prime_arrays = rows, arrays
            elif label == 'prime_on':
                differences = {}
                assert rows.keys() == prime_rows.keys(), 'old nested carry structure changed'
                for key, value in arrays.items():
                    a = prime_arrays[key]
                    if rows[key] != prime_rows[key]:
                        assert a.shape == value.shape and a.dtype == value.dtype
                        delta = a.astype(np.float64)-value.astype(np.float64)
                        differences[key] = dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta*delta))))
                arm['old_carry_differences'] = differences
                prime_arrays.clear(); arrays.clear()
                if differences:
                    record['phases'][label] = arm; record['verdict'] = 'STOP_NONBYTE_PRIME'
                    atomic(output,record); raise SystemExit(2)
            elif label == 'replay_off':
                assert len(files) == 9, files
                reference_rows, reference_files = rows, files
            else:
                assert len(files) == 9, files
                arm['terminal_byte_equal'] = rows == reference_rows
                arm['wrfout_byte_equal'] = files == reference_files
                if not arm['terminal_byte_equal'] or not arm['wrfout_byte_equal']:
                    record['phases'][label] = arm; record['verdict'] = 'STOP_NONBYTE_3H'
                    atomic(output,record); raise SystemExit(2)
            proof = out/'wrfout/proofs/nested_pipeline_run.json'
            if proof.exists():
                arm['pipeline_proof_sha256'] = sha(proof)
                arm['aot'] = json.loads(proof.read_text()).get('metadata',{}).get('nested_aot')
            else:
                arm['aot'] = current.get('aot')
            record['phases'][label] = arm
            atomic(output,record)
            print(label, json.dumps(dict(wall_s=arm['wall_s'], files=arm['files'])), flush=True)
    finally:
        pipeline._load_domains, pipeline.run_operational_domain_tree = original_load, original_segment
    record['verdict'] = 'FIRST_STEP_AND_3H_BYTE'
    atomic(output,record)


if __name__ == '__main__':
    main()
