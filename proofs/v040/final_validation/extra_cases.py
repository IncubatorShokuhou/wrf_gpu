"""Authentic SSiB real.exe inputs, K1 composite and full WN3 two-way reference."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare import ROOT, RETAINED, PRISTINE, SLOTS, private_executable, render, sha, write_json
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.namelist_binding import namelist_honour_report


def stage(row, base, groups, extra_files=None):
    case = ROOT / row['id']
    assert not (case / 'cpu_run.json').exists(), 'Never rewrite a measured reference'
    inp, cpu = case / 'input', case / 'cpu'
    inp.mkdir(parents=True, exist_ok=True)
    cpu.mkdir(exist_ok=True)
    (inp / 'namelist.input').write_text(render(groups))
    names = ['wrfbdy_d01'] + ['wrfinput_' + d for d in row['domains']]
    for name in names:
        if not (inp / name).exists():
            (inp / name).symlink_to((base / name).resolve())
    for name, data in (extra_files or {}).items():
        (inp / name).write_text(data)
    errors, warnings = namelist_honour_report(groups, tuple(row['domains']), replay=False)
    row.update(admission_errors=errors, admission_warnings=warnings,
               status='ADMITTED' if not errors else 'REFUSED')
    private_executable(case)
    for item in (PRISTINE / 'run').iterdir():
        if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
            if not (cpu / item.name).exists():
                (cpu / item.name).symlink_to(item.resolve())
    for item in inp.iterdir():
        if (cpu / item.name).is_symlink():
            (cpu / item.name).unlink()
        (cpu / item.name).symlink_to(item.resolve())
    if not (cpu / 'wrf.exe').exists():
        (cpu / 'wrf.exe').symlink_to(case / 'wrf.exe')
    if row['options']['mp_physics'] == 8:
        for table in ('qr_acr_qg_V4.dat', 'qr_acr_qsV2.dat'):
            p = cpu / table
            if not p.exists():
                p.symlink_to(RETAINED / 'bl9/cpu' / table)
    row['input_sha256'] = {p.name: sha(p) for p in inp.iterdir() if p.is_file()}
    write_json(case / 'inputs.json', row)
    return row


def ssib():
    work = ROOT / 'ssib_real'
    work.mkdir(exist_ok=True)
    base = RETAINED / 'urban_madrid/input'
    groups = parse_namelist(base / 'namelist.input')
    groups['physics'].update(sf_surface_physics=8, num_soil_layers=3, sf_urban_physics=0,
                             bl_pbl_physics=1, sf_sfclay_physics=1, ra_lw_physics=4,
                             ra_sw_physics=1, cu_physics=0, fractional_seaice=1)
    (work / 'namelist.input').write_text(render(groups))
    for p in base.glob('met_em.*'):
        if not (work / p.name).exists():
            (work / p.name).symlink_to(p.resolve())
    for p in (PRISTINE / 'run').iterdir():
        if p.is_file() and p.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
            if not (work / p.name).exists():
                (work / p.name).symlink_to(p.resolve())
    with (work / 'real.log').open('w') as log:
        r = subprocess.run(['timeout', '300', str(PRISTINE / 'main/real.exe')], cwd=work,
                           stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(rc=r.returncode, executable_sha256=sha(PRISTINE / 'main/real.exe'),
        namelist_sha256=sha(work / 'namelist.input'),
        met_em_sha256={p.name: sha(p) for p in work.glob('met_em.*')},
        success='SUCCESS COMPLETE REAL' in (work / 'real.log').read_text(errors='replace'),
        geography='retained authentic Madrid WPS met_em, no fabricated soil interpolation')
    write_json(work / 'receipt.json', receipt)
    assert r.returncode == 0 and receipt['success'], receipt
    row = dict(id='c40', geography='Madrid-SSiB', hours=24, domains=['d01'],
               options={k: groups['physics'][k] for k in SLOTS}, extra={}, features=[])
    return stage(row, work, groups)


def k1():
    base = RETAINED / 'source_inputs/Swiss'
    groups = parse_namelist(Path(sys.argv[1]) / 'examples/switzerland_d01/namelist.input')
    groups['physics'].update(cu_physics=1, prec_acc_dt=60., windfarm_opt=1,
                             windfarm_ij=1, windfarm_tke_factor=.25, do_radar_ref=1)
    groups['time_control'].update(nwp_diagnostics=1)
    groups['dynamics'].update(tracer_opt=2)
    groups['stoch'] = dict(skebs=1, sppt=1, spp_pbl=1, rand_perturb=1, nens=7)
    streams = dict(version=1, streams=[dict(name='validation', domain='d01', interval_steps=200,
        buffer_frames=4, region=dict(stations=[dict(id='centre', i=20, j=20)]),
        variables=dict(T2=['instant'], RAINNC=['instant']))])
    files = {'tslist': (Path(sys.argv[1]) / 'proofs/u2_tslist/oracle/tslist').read_text(),
        'gpuwrf.streams.json': json.dumps(streams, indent=2) + '\n',
        'windturbines-ij.txt': '21 21 1\n22 21 1\n',
        'wind-turbine-1.tbl': '6\n80 80 .12 2000\n.5 .9 0\n2 .85 20\n8 .8 500\n12 .7 1600\n20 .4 2000\n25 .1 2000\n'}
    row = dict(id='c41', geography='Swiss', hours=24, domains=['d01'],
        options={k: groups['physics'][k] for k in SLOTS}, extra={},
        features=['tslist', 'nwp_diagnostics', 'prec_acc_dt=60', 'regional_streams',
                  'WRF-tracer_opt=2', 'Fitch-two-declared-turbines', 'SKEBS/SPPT/SPP-PBL/RAND'])
    return stage(row, base, groups, files)


def nested():
    base = Path('<USER_HOME>/wrf_gpu2_lanes/u2-twoway/cpu-reference')
    groups = parse_namelist(base / 'namelist.input')
    tc = groups['time_control']
    tc.update(run_days=0, run_hours=24, run_minutes=0, run_seconds=0,
        end_year=[2026, 2026], end_month=[3, 3], end_day=[1, 1],
        end_hour=[0, 0], end_minute=[0, 0], end_second=[0, 0],
        history_interval=[60, 60], history_interval_s=[0, 0])
    for group in groups.values():
        for key, value in group.items():
            if isinstance(value, list):
                group[key] = value[:2]
    groups['physics'].update(slope_rad=[0, 0], topo_shading=[0, 0], gwd_opt=[0, 0])
    # gwd_opt belongs to dynamics, not physics; preserve the WRF group authority.
    groups['physics'].pop('gwd_opt')
    groups['dynamics']['gwd_opt'] = [0, 0]
    row = dict(id='c42', geography='WN3-20260227_18z+6h', hours=24, domains=['d01', 'd02'],
        options={k: groups['physics'][k][0] if isinstance(groups['physics'][k], list) else groups['physics'][k] for k in SLOTS},
        extra={'domains': {'feedback': 1, 'smooth_option': 2}}, features=['live-two-way-feedback'],
        geometry='full authentic WN3 d01 120x70 + d02 267x117, 44 levels; no crop')
    return stage(row, base, groups)


def surface7():
    base = RETAINED / 'source_inputs/Swiss'
    groups = parse_namelist(Path(sys.argv[1]) / 'examples/switzerland_d01/namelist.input')
    groups['physics'].update(bl_pbl_physics=0, sf_sfclay_physics=7)
    row = dict(id='c43', geography='Swiss', hours=24, domains=['d01'],
        options={k: groups['physics'][k] for k in SLOTS}, extra={},
        features=['Pleim-Xiu-surface-layer-with-native-NoahMP'])
    return stage(row, base, groups)


def lake():
    base = RETAINED / 'source_inputs/Swiss'
    groups = parse_namelist(Path(sys.argv[1]) / 'examples/switzerland_d01/namelist.input')
    groups['physics'].update(sf_lake_physics=1)
    row = dict(id='c44', geography='Swiss', hours=24, domains=['d01'],
        options={k: groups['physics'][k] for k in SLOTS}, extra={'physics': {'sf_lake_physics': 1}},
        features=['lake1-active-geography-census-required'])
    return stage(row, base, groups)


def k1_resolved():
    base = ROOT / 'c41/input'
    groups = parse_namelist(base / 'namelist.input')
    # Match the independent stochastic lane's resolved small-domain profile.
    # Registry PBL700km/RAND500km Gaussian scales cannot be assumed active on
    # a126km Swiss domain. Retain the original default-scale CPU row as a
    # conditioning/inactivity control, never spend a GPU compile on it.
    for name in ('sppt', 'rand_pert', 'spp_conv', 'spp_pbl', 'spp_lsm'):
        groups['stoch']['lengthscale_' + name] = 15000.
    groups['stoch'].update(spp=1,spp_conv=1,spp_lsm=1,stoch_force_opt=1)
    files = {p.name: p.read_text() for p in base.iterdir()
             if p.is_file() and p.name not in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01')}
    row = dict(id='c46', geography='Swiss', hours=24, domains=['d01'],
        options={k: groups['physics'][k] for k in SLOTS}, extra={},
        features=['tslist', 'nwp_diagnostics', 'prec_acc_dt=60', 'regional_streams',
                  'WRF-tracer_opt=2', 'Fitch-two-declared-turbines', 'SKEBS/SPPT/SPP-PBL/RAND',
                  'resolved-Gaussian15km','SPP-master+conv+LSM-original-no-op-consumers',
                  'SKEBS-alias-stoch_force_opt1'])
    return stage(row, base, groups, files)


if __name__ == '__main__':
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    matrix = json.loads((ROOT / 'MATRIX.json').read_text())
    for factory in (ssib, k1, nested, surface7, lake, k1_resolved):
        existing = {'ssib': 'c40', 'k1': 'c41', 'nested': 'c42', 'surface7': 'c43', 'lake': 'c44', 'k1_resolved': 'c46'}[factory.__name__]
        if any(r['id'] == existing for r in matrix['cases']):
            continue
        row = factory()
        matrix['cases'].append(row)
        if factory is k1_resolved:
            old = next(r for r in matrix['cases'] if r['id'] == 'c41')
            old.update(gpu_eligible=False, status='DEFAULT_GAUSSIAN_CONTROL_ONLY', superseded_by='c46')
            write_json(ROOT / 'c41/inputs.json', old)
        write_json(ROOT / 'MATRIX.json', matrix)
        print(json.dumps({'case': row['id'], 'errors': row['admission_errors']}), flush=True)
