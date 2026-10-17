"""Compare the dedicated CuP parcel with the full pristine PARA boundary.

CPU Pallas interpretation establishes numerical evidence only. GPU compile,
launch count, time and memory must be measured separately in a manager window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def parse(path):
    return parse_text(Path(path).read_text())


def parse_text(text):
    profiles, scalars = {}, {}
    for line in text.splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        try:
            value = float(value)
        except ValueError:
            continue
        if '[' in key:
            name, index = key[:-1].split('[')
            profiles.setdefault(name, {})[int(index)] = value
        else:
            scalars[key] = value
    return {name: [values[k] for k in sorted(values)] for name, values in profiles.items()}, scalars


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--cps', action='store_true')
    args = p.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0, str(args.source / 'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.kernels.phys_kfcup_column import kfcup_parcel
    if args.cps:
        from gpuwrf.kernels.phys_kfcup_cps import kfcup_fresh_column

    assert all(device.platform == 'cpu' for device in jax.devices())
    gate = json.loads((args.source / 'proofs/v040/cuplus/KFCUP_PARCEL_GATE.json').read_text())
    oracle = json.loads(args.oracle.read_text())
    profiles_in = ('T', 'QV', 'P', 'DZ', 'RHO', 'W0AVG', 'U', 'V')
    tendency = ('RTHCUTEN', 'RQVCUTEN', 'RQCCUTEN', 'RQRCUTEN', 'RQICUTEN', 'RQSCUTEN')
    record = dict(backend=jax.default_backend(), source=str(args.source), oracle=str(args.oracle),
                  gate=gate, rows=[], stage='STARTED', gpu_claim=False)

    def one(*values):
        if args.cps:
            return kfcup_fresh_column(*values[:12],dt=values[12][0],dx=values[12][1],
                                      pblh=values[12][2],kpbl=values[12][3],hfx=values[12][4],qfx=values[12][5],
                                      land=values[12][6],terrain=values[12][7],num_bins=5,interpret=True)
        return kfcup_parcel(*values[:8], values[8][0], values[8][1], 40,
                            cupflag=values[8][2], th_perturb=values[8][3], r_perturb=values[8][4],
                                freq=values[8][5], pblh=values[8][6], z_at_w=values[9], interpret=True, trace=args.trace)

    fn = jax.jit(one)
    saved = {}
    start = time.monotonic()
    for row in oracle['records']:
        if row['mode'] != 'plain':
            continue
        reference, scalars = parse(row['stdout'])
        assert row['returncode'] == 0 and row['output']['finite'], row
        assert all(np.isfinite(value).all() for value in reference.values())
        assert all(np.isfinite(value) for value in scalars.values())
        dz = np.asarray(reference['DZ'], np.float32)
        zw = np.concatenate([np.zeros(1, np.float32), np.cumsum(dz[:-1], dtype=np.float32)])
        if args.cps:
            controls=jnp.asarray([scalars['DT'],scalars['DX'],scalars.get('CTRL_PBLH',1500.),
                scalars.get('CTRL_KPBL',6.),scalars.get('CTRL_HFX',float(row['controls'][0]) if row['controls'] else 0.),
                scalars.get('CTRL_QFX',1e-4),scalars.get('CTRL_XLAND',1.),scalars.get('CTRL_HT',0.)],jnp.float32)
            inputs=tuple(jnp.asarray(reference[key],jnp.float32) for key in profiles_in+('TH','Z','ZW','PI'))+(controls,)
        else:
            controls = jnp.asarray([scalars['DT'], scalars['DX'], scalars['CUPFLAG'],
                                    scalars['TH_PERTURB'], scalars['R_PERTURB'], scalars['FREQ'],
                                    float(row['controls'][1])], jnp.float32)
            inputs = tuple(jnp.asarray(reference[key], jnp.float32) for key in profiles_in) + (controls, jnp.asarray(zw))
        begun = time.monotonic()
        actual = {key: np.asarray(value) for key, value in jax.block_until_ready(fn(*inputs)).items()}
        for key,value in actual.items():
            saved[f'case{row["case"]}_{row["label"]}_{key}'] = value
        np.savez(args.output.with_suffix('.npz'), **saved)
        comparison = dict(case=row['case'], label=row['label'], fields={}, scalars={}, failures=[],
                          cpu_elapsed_s=time.monotonic()-begun,
                          candidate_nonfinite=int(sum(np.count_nonzero(~np.isfinite(x)) for x in actual.values())))
        for key, target in reference.items():
            if key not in actual:
                continue
            target = np.asarray(target, np.float32)
            value = actual[key][:len(target)]
            error = float(np.max(np.abs(value-target)))
            scale = float(np.max(np.abs(target)))
            tol = gate['tendencies' if key in tendency else 'profiles']
            limit = max(tol['absolute'], tol['column_peak_relative'] * scale)
            passed = bool(np.isfinite(value).all() and error <= limit)
            comparison['fields'][key] = dict(max_abs=error, peak=scale, limit=limit, pass_=passed)
            if not passed:
                comparison['failures'].append(key)
        scalar_keys = (('RAINCV','PRATEC','NCA','CUTOP','CUBOT','SHALL','CUPFLAG','ACTIVEFRAC','TAUCLOUD','TACTIVE','LNINT')
                       if args.cps else ('RAINCV','PRATEC','NCA','CUTOP','CUBOT','SHALL','UPDFRA','WLCL','WULCL'))
        for key in scalar_keys:
            if key not in scalars:
                continue
            value = float(actual['ISHALL' if key == 'SHALL' else key])
            target = scalars[key]
            error = abs(value-target)
            tol = gate['raincv' if key == 'RAINCV' else 'scalar_float']
            limit = 0.0 if key in gate['exact'] else max(tol['absolute'], tol['relative']*abs(target))
            passed = bool(np.isfinite(value) and error <= limit)
            comparison['scalars'][key] = dict(actual=value, wrf=target, max_abs=error, limit=limit, pass_=passed)
            if not passed:
                comparison['failures'].append(key)
        record['rows'].append(comparison)
        record['stage'] = f'COMPARED {len(record["rows"])}'
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps({key: comparison[key] for key in ('case','label','failures','cpu_elapsed_s','candidate_nonfinite')}), flush=True)
    record['elapsed_s'] = time.monotonic()-start
    record['stage'] = 'COMPLETE'
    record['verdict'] = 'PASS' if all(not row['failures'] and not row['candidate_nonfinite'] for row in record['rows']) else 'FAIL'
    record['kernel_sha256'] = hashlib.sha256((args.source/'src/gpuwrf/kernels/phys_kfcup_column.py').read_bytes()).hexdigest()
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(record['verdict'], flush=True)
    return 0 if record['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
