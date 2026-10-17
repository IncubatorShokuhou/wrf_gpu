"""CPU-only preflight of writer metadata against every original WRF frame.

The declared-spec audit is deliberately an upper bound: a declared variable
without a producer is still reported separately by the emission gate.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS, _dimension_sizes


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit(root: Path, output: Path):
    matrix = json.loads((root / 'MATRIX.json').read_text())
    rows = []
    for case in matrix['cases']:
        folder = root / case['id']
        raw = parse_namelist(folder / 'input/namelist.input')
        for domain in case['domains']:
            idx = int(domain[1:]) - 1
            def get(section, key, default=0):
                v = raw.get(section, {}).get(key, default)
                return v[min(idx, len(v)-1)] if isinstance(v, (tuple, list)) else v
            nl = {k: get('physics', k) for k in case['options']}
            nl['num_soil_layers'] = get('physics', 'num_soil_layers', 4)
            if nl['sf_surface_physics'] == 3:
                nl['ruc_static'] = SimpleNamespace(cfg=SimpleNamespace(nzs=nl['num_soil_layers']))
            files = sorted((folder / 'cpu').glob(f'wrfout_{domain}_*'))
            row = dict(case=case['id'], domain=domain, status=case['status'], frames=len(files),
                       missing_specs=[], metadata={}, dimensions={}, globals={}, cpu_inventory={})
            attrs = wrf_history_global_attributes(folder / f'input/wrfinput_{domain}', raw, domain,
                                                 float(get('domains', 'time_step', 18)))
            for frame, path in enumerate(files):
                with ExitStack() as stack:
                    try:
                        ds = stack.enter_context(Dataset(path))
                    except OSError as exc:
                        row.setdefault('unreadable', []).append(dict(path=str(path), error=str(exc)))
                        continue
                    dims = {k: len(v) for k, v in ds.dimensions.items()}
                    planned = _dimension_sizes(nx=dims['west_east'], ny=dims['south_north'],
                                               nz=dims['bottom_top'], namelist=nl)
                    for k, v in dims.items():
                        if k != 'Time' and planned.get(k) != v:
                            row['dimensions'].setdefault(k, dict(cpu=v, planned=planned.get(k), frames=[]))['frames'].append(frame)
                    # Child dt comes from resolved parent ratio, independently of
                    # the reference's observed DT.
                    if idx:
                        ratios = raw['domains'].get('parent_time_step_ratio', [1])
                        attrs['DT'] = np.float32(get('domains', 'time_step', 18) / np.prod(ratios[1:idx+1]))
                    for k in ds.ncattrs():
                        if k in {'TITLE', 'NTASKS_X', 'NTASKS_Y', 'NTASKS_TOTAL'}:
                            continue
                        v = ds.getncattr(k)
                        if k not in attrs or not np.array_equal(attrs[k], v):
                            row['globals'].setdefault(k, dict(cpu=plain(v), planned=plain(attrs.get(k)), frames=[]))['frames'].append(frame)
                    for k, v in ds.variables.items():
                        inv = dict(dimensions=list(v.dimensions), dtype=str(v.dtype), attrs={
                            a: plain(v.getncattr(a)) for a in v.ncattrs()})
                        row['cpu_inventory'].setdefault(k, inv)
                        if k == 'Times':
                            continue
                        spec = WRFOUT_VARIABLE_SPECS.get(k)
                        if spec is None:
                            if k not in row['missing_specs']:
                                row['missing_specs'].append(k)
                            continue
                        expected = dict(dimensions=list(spec.dimensions), dtype=str(np.dtype(spec.dtype)),
                                        MemoryOrder=spec.memory_order, description=spec.description,
                                        units=spec.units, stagger=spec.stagger, coordinates=spec.coordinates)
                        observed = dict(dimensions=list(v.dimensions), dtype=str(v.dtype), **inv['attrs'])
                        for a, want in expected.items():
                            if k == 'XTIME' and a in ('units', 'description'):
                                continue  # run-start-dependent text checked by emission tests
                            if observed.get(a) != want:
                                row['metadata'].setdefault(f'{k}.{a}', dict(cpu=observed.get(a), planned=want, frames=[]))['frames'].append(frame)
            rows.append(row)
            output.write_text(json.dumps(dict(schema='t-writer-declared-metadata-v1', rows=rows), indent=2) + '\n')
            print(case['id'], domain, len(files), 'dims', len(row['dimensions']), 'globals', len(row['globals']),
                  'metadata', len(row['metadata']), 'missing_specs', row['missing_specs'], flush=True)
    return dict(schema='t-writer-declared-metadata-v1', utc=datetime.now(timezone.utc).isoformat(),
                scope='declared metadata upper bound; does not attest producer availability or emitted inventory', rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.root, a.output), indent=2) + '\n')
