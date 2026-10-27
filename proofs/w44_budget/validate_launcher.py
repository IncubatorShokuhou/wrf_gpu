"""E225 launcher check: REAL `gpuwrf run --dry-run` with the launcher's exact CLI argv + explicit alignments.

usage: JAX_PLATFORMS=cpu python validate_launcher.py MANIFEST OUT_JSON
The argv is rebuilt exactly as night_probe_design_b.py builds it (cli.main(['run', ...])), plus --dry-run.
"""
import json, subprocess, sys, tempfile, hashlib
from pathlib import Path
manifest_path, out = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads(manifest_path.read_text()); model = Path(m['model']); case = Path(m['case'])
duration = m.get('duration_args', ['--hours', '3']); max_dom = m.get('max_dom', 3)
sys.path.insert(0, str(model / 'src'))
from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.runtime.run_window import aligned_steps
nl = parse_namelist(case / 'namelist.input')
def lst(group, key):
    v = nl.get(group, {}).get(key); return list(v) if isinstance(v, (list, tuple)) else [v]
root_dt = float(lst('domains', 'time_step')[0]); ratios = [int(x) for x in lst('domains', 'parent_time_step_ratio')][:max_dom]
dts = [root_dt]
for r in ratios[1:]: dts.append(dts[-1] / r)  # WN3 chain d01->d02->d03 (parent_id 1,2)
hours = float(duration[1]) if duration[0] == '--hours' else float(duration[1]) / 60
seconds = hours * 3600
hist = [float(x) * 60 for x in lst('time_control', 'history_interval')][:max_dom]
bdy = float(lst('time_control', 'interval_seconds')[0])
checks = {}
info = {}
def check(name, ok, detail): checks[name] = dict(ok=bool(ok), detail=detail)
try: aligned_steps(hours, root_dt); check('cli_aligned_steps_root', True, f'{hours} h / {root_dt} s')
except ValueError as e: check('cli_aligned_steps_root', False, str(e))
for i, dt in enumerate(dts):
    check(f'duration_div_dt_d0{i+1}', abs(seconds / dt - round(seconds / dt)) < 1e-9, f'{seconds} s / {dt} s = {seconds/dt}')
    info[f'history_div_dt_d0{i+1}'] = f'{hist[i]} s / {dt} s = {hist[i]/dt} (WRF alarm rings at the first step at/after the interval, e.g. 3618 s on WN3 d01; informational)'
    check(f'duration_div_history_d0{i+1}', abs(seconds / hist[i] - round(seconds / hist[i])) < 1e-9, f'{seconds} s / {hist[i]} s')
check('boundary_interval_div_root_dt', abs(bdy / root_dt - round(bdy / root_dt)) < 1e-9, f'{bdy} s / {root_dt} s')
check('wn3_hours_multiple_of_3_or_short_window', duration[0] != '--hours' or int(hours) % 3 == 0, f'{duration}')
frames = int(round(seconds / hist[0])) * max_dom
check('expected_history_files', frames == m.get('expected_history_files', 9), f'{frames} vs manifest {m.get("expected_history_files", 9)}')
with tempfile.TemporaryDirectory(dir='<USER_HOME>/wrf_gpu2_lanes/w44-budget') as tmp:
    argv = ['run', '--input-dir', str(case), '--output-dir', str(Path(tmp) / 'wrfout'),
            '--scratch-dir', str(Path(tmp) / 'scratch'), *duration, '--max-dom', str(max_dom), '--dry-run']
    proc = subprocess.run([sys.executable, '-c', 'import sys;from gpuwrf import cli;raise SystemExit(cli.main(sys.argv[1:]))', *argv],
        env=dict(__import__('os').environ, PYTHONPATH=str(model / 'src'), JAX_PLATFORMS='cpu', GPUWRF_FAST_DEFAULTS='1', GPUWRF_MYNN_EDMF='1'),
        capture_output=True, text=True, timeout=600)
check('cli_dry_run_rc0', proc.returncode == 0, f'rc={proc.returncode}')
try: plan = json.loads(proc.stdout[proc.stdout.index('{'):])
except ValueError: plan = None
result = dict(manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    model=str(model), model_pin=m['model_pin'], case=str(case), argv=argv, dts=dts, history_s=hist, boundary_s=bdy,
    duration_s=seconds, checks=checks, info=info, dry_run_stderr_tail=proc.stderr[-2000:],
    plan_hours=(plan or {}).get('hours') or (plan or {}).get('effective_hours'), plan_keys=sorted((plan or {}).keys())[:60],
    verdict='VALIDATED' if all(c['ok'] for c in checks.values()) else 'FAIL')
out.write_text(json.dumps(result, indent=2) + '\n'); print(result['verdict'], {k: v['ok'] for k, v in checks.items()})
raise SystemExit(0 if result['verdict'] == 'VALIDATED' else 2)
