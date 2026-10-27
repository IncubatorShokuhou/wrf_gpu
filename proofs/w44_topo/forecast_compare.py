"""Quantify old/new forecast effect; GPU-vs-GPU is NOT a fidelity reference."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile

from netCDF4 import Dataset
import numpy as np


def statistics(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('forecast effect shape/dtype mismatch')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('forecast effect rejects nonfinite old AND new arrays')
    delta = right.astype(np.float64) - left.astype(np.float64)
    return dict(count=int(delta.size), unequal=int(np.count_nonzero(left != right)),
                max_abs=float(np.max(np.abs(delta))), sum=float(delta.sum()),
                mean=float(delta.mean()), old_max=float(left.max()), new_max=float(right.max()),
                sum_squared=float(np.sum(delta * delta)),
                rmse=float(np.sqrt(np.mean(delta * delta))),
                byte_identical=left.tobytes() == right.tobytes())


def compare(old, new, expected=36, hours=11, report_start=8):
    old, new = Path(old), Path(new)
    left = {p.name: p for p in (old / 'wrfout').glob('wrfout_d0*')}
    right = {p.name: p for p in (new / 'wrfout').glob('wrfout_d0*')}
    if len(left) != expected or set(left) != set(right):
        raise ValueError(f'incomplete/misaligned forecast coverage {len(left)}/{len(right)}')
    full = expected == 3 * (hours + 1)
    if full:
        coverage = {d: [] for d in ('d01', 'd02', 'd03')}
        for name in sorted(left):
            domain = name.split('_')[1]
            valid = datetime.strptime(name.split('_', 2)[2], '%Y-%m-%d_%H:%M:%S')
            if valid.date().isoformat() != '2026-02-28':
                raise ValueError('unexpected forecast date')
            coverage[domain].append(valid.hour)
        if any(seen != list(range(hours+1)) for seen in coverage.values()):
            raise ValueError(f'incomplete 00-11Z hourly coverage {coverage}')
    rows = []
    for name in sorted(left):
        with Dataset(left[name]) as a, Dataset(right[name]) as b:
            a.set_auto_mask(False); b.set_auto_mask(False)
            fields = {}
            for field in ('SWDOWN', 'T2'):
                if field not in a.variables or field not in b.variables:
                    raise ValueError(f'missing {name}:{field}')
                fields[field] = statistics(a[field][:], b[field][:])
            if 'GSW' in a.variables and 'GSW' in b.variables:
                fields['GSW'] = statistics(a['GSW'][:], b['GSW'][:])
            # WRF history field of the slope/shadow-adjusted surface SW (rhd).
            if 'SWNORM' in a.variables and 'SWNORM' in b.variables:
                fields['SWNORM'] = statistics(a['SWNORM'][:], b['SWNORM'][:])
        rows.append(dict(frame=name, domain=name.split('_')[1], fields=fields))
    lf = {p.name: p for p in (old / 'held_surface').glob('d0*.npz')}
    rf = {p.name: p for p in (new / 'held_surface').glob('d0*.npz')}
    if len(lf) != expected or set(lf) != set(rf):
        raise ValueError('missing/alignment failure in held GSW output-boundary capture')
    net_rows = []
    for name in sorted(lf):
        if full and 'wrfout_' + Path(name).stem not in left:
            raise ValueError('held GSW timestamp does not match a history frame')
        with np.load(lf[name]) as a, np.load(rf[name]) as b:
            net_rows.append(dict(frame=name, domain=name.split('_')[0],
                                 GSW=statistics(a['GSW'], b['GSW'])))
    for run in (old, new):
        receipt = json.loads((run / 'receipt.json').read_text())
        proof = json.loads((run / 'cli_stdout.json').read_text())
        if receipt.get('rc') != 0 or receipt.get('device', {}).get('platform') != 'gpu':
            raise ValueError('forecast process was not a successful asserted GPU run')
        if not proof.get('all_domains_finite') or not proof.get('all_outputs_present'):
            raise ValueError('forecast finite/output gate incomplete')
    if full:
        rows = [r for r in rows if report_start <= datetime.strptime(r['frame'].split('_', 2)[2], '%Y-%m-%d_%H:%M:%S').hour <= hours]
        net_rows = [r for r in net_rows if report_start <= datetime.strptime(Path(r['frame']).stem.split('_', 1)[1], '%Y-%m-%d_%H:%M:%S').hour <= hours]
        wanted = 3 * (hours-report_start+1)
        if len(rows) != wanted or len(net_rows) != wanted:
            raise ValueError('incomplete daylight 08-11Z slice')
        if report_start >= 8 and not all(any(r['fields']['SWDOWN'][side + '_max'] > 0 for r in rows) for side in ('old', 'new')):
            raise ValueError('daylight slice contains no positive SWDOWN')
    aggregates = {}
    for domain in sorted({r['domain'] for r in rows}):
        aggregates[domain] = {}
        for field in ('SWDOWN', 'SWNORM', 'T2', 'GSW'):
            samples = [r['fields'][field] for r in rows if r['domain'] == domain and field in r['fields']]
            if field == 'GSW':
                samples = [r['GSW'] for r in net_rows if r['domain'] == domain]
            if not samples:
                continue
            count = sum(r['count'] for r in samples)
            aggregates[domain][field] = dict(count=count, unequal=sum(r['unequal'] for r in samples),
                max_abs=max(r['max_abs'] for r in samples), mean=sum(r['sum'] for r in samples)/count,
                rmse=float(np.sqrt(sum(r['sum_squared'] for r in samples)/count)))
    return dict(status='PASS', scope='GPU-vs-GPU forecast effect/regression only; NOT CPU-WRF fidelity',
                finite_checked_frames=expected, reported_slice=f'{report_start:02d}-{hours:02d}Z after unchanged 00Z initialization',
                frames=len(rows), rows=rows, held_GSW=net_rows, aggregates=aggregates,
                GSW_definition='slope-adjusted net SW (LSM forcing) = held swnorm minus swup_topographic, recovered on host at output only; not WRF history GSW',
                new_WN3_programs=json.loads((new / 'cli_stdout.json').read_text()).get('metadata', {}).get('nested_aot'))


def selftest(root):
    for side in ('old', 'new'):
        folder=root/side
        (folder/'wrfout').mkdir(parents=True);(folder/'held_surface').mkdir()
        with Dataset(folder/'wrfout/wrfout_d01_test','w') as ds:
            ds.createDimension('n',2)
            for name in ('SWDOWN','SWNORM','T2'):
                ds.createVariable(name,'f4',('n',))[:]=[1,2]
        np.savez(folder/'held_surface/d01_00000001.npz',GSW=np.array([1,2],np.float32))
        (folder/'receipt.json').write_text(json.dumps({'rc':0,'device':{'platform':'gpu'}}))
        (folder/'cli_stdout.json').write_text(json.dumps({'all_domains_finite':True,'all_outputs_present':True}))
    first=compare(root/'old',root/'new',1)
    assert first['rows'][0]['fields']['T2']['byte_identical'] and 'SWNORM' in first['aggregates']['d01']
    with Dataset(root/'new/wrfout/wrfout_d01_test','a') as ds:
        ds['T2'][0]=np.nextafter(np.float32(1),np.float32(2))
    assert compare(root/'old',root/'new',1)['rows'][0]['fields']['T2']['unequal']==1
    for value in (np.nan,np.inf,-np.inf):
        for side in ('old','new'):
            with Dataset(root/side/'wrfout/wrfout_d01_test','a') as ds:ds['T2'][0]=value
            try:compare(root/'old',root/'new',1)
            except ValueError as exc:assert 'nonfinite' in str(exc)
            else:raise AssertionError('nonfinite forecast effect input survived')
            with Dataset(root/side/'wrfout/wrfout_d01_test','a') as ds:ds['T2'][0]=1
    print('forecast effect reader PASS: baseline,1ULP,old/new NaN/Inf controls')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--old',type=Path);p.add_argument('--new',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--selftest',type=Path)
    p.add_argument('--hours',type=int,default=11);p.add_argument('--report-start',type=int,default=8)
    p.add_argument('--launch-check',type=Path)
    args=p.parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory(dir=args.selftest) as temp:selftest(Path(temp))
    elif args.launch_check:
        records=[]
        native=json.loads((args.launch_check/'native/operator/bench.json').read_text())
        assert native['status']=='CPU_LAUNCH_SMOKE_PASS' and len(native['rows'])==24
        for name in ('old','new'):
            root=args.launch_check/name
            model=json.loads((root/'LAUNCH_SMOKE_MODEL.json').read_text())
            env=json.loads((root/'DEVICE_ENV.json').read_text())
            assert model['status']=='CPU_LAUNCH_SMOKE_PASS' and model['finite_leaves']>0 and model['nonfinite_leaves']==0
            assert model['model_hook']=='real (no substitution)' and model['stop_point'].startswith('post-init')
            assert model['domains']==['d01','d02','d03'],model['domains']
            if name=='new':
                # F1 on the real case: every nest starts from float32 allocator-zero HGT_SHAD.
                for domain,height in model['shadow_height'].items():
                    assert height and height['dtype']=='float32' and height['all_zero'] and height['finite'],(domain,height)
            assert env['product_env']['JAX_PLATFORMS']=='cuda'
            records.append(dict(phase=name,model=model,device_env=env))
        # E41 verdicts (E230: parsed JSON, INCOMPLETE allowed) are validated by the driver, after this check.
        args.out.write_text(json.dumps(dict(status='CPU_ALL_PHASES_PASS',scope='real CPU model calls; NOT CUDA mini qualification',native_cases=len(native['rows']),rows=records,GPU_started=False),indent=2)+'\n')
    else:args.out.write_text(json.dumps(compare(args.old,args.new,3*(args.hours+1),args.hours,args.report_start),indent=2)+'\n')
