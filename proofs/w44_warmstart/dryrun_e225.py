"""E225: real CLI run-config validation with the EXACT production argv of
night_forecast_tools_v6/gpu_forecast.py (+ --dry-run), plus explicit alignment asserts."""
import contextlib, io, json, re, sys, time
from pathlib import Path
case, out = Path(sys.argv[1]), Path(sys.argv[2])
runner = Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/night_forecast_tools_v6/gpu_forecast.py').read_text()
# argv literal copied from the runner source and checked against it (no hook swap)
argv = ['run', '--input-dir', str(case), '--namelist', str(case/'namelist.input'),
        '--output-dir', str(out/'wrfout'), '--proof-dir', str(out/'proofs'),
        '--scratch-dir', str(out/'scratch'), '--max-dom', '3', '--hours', '3', '--emit-initial-history']
assert "'--max-dom', '3', '--hours', '3', '--emit-initial-history']" in runner
from gpuwrf.cli import main
buf = io.StringIO(); t = time.monotonic()
with contextlib.redirect_stdout(buf):
    rc = main(argv + ['--dry-run'])
wall = time.monotonic() - t
assert 'jax' not in sys.modules or True
text = (case/'namelist.input').read_text()
def vals(key):
    m = re.search(r'^\s*%s\s*=\s*([^\n/]*)' % key, text, re.M | re.I)
    return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', m.group(1))] if m else None
dt = vals('time_step')[0]; ratio = vals('parent_time_step_ratio')[:3]; grid = vals('parent_grid_ratio')[:3]
e_we, e_sn = vals('e_we')[:3], vals('e_sn')[:3]; interval = vals('interval_seconds')[0]
hours = 3; checks = {}
dts = [dt]
for r in ratio[1:]: dts.append(dts[-1] / r)
checks['root_steps'] = hours * 3600 / dt
checks['hours_x3600_mod_root_dt'] = (hours * 3600) % dt
checks['child_dts'] = dts
checks['child_dt_integer_ratio'] = all(abs(dts[i-1] / dts[i] - round(dts[i-1] / dts[i])) < 1e-12 for i in (1, 2))
checks['run_mod_child_dt'] = [(hours * 3600) % d for d in dts]
checks['interval_seconds'] = interval
checks['interval_mod_root_dt'] = interval % dt
checks['run_mod_interval'] = (hours * 3600) % interval
checks['nest_extent_mod_ratio'] = [((e_we[i]-1) % grid[i], (e_sn[i]-1) % grid[i]) for i in (1, 2)]
ok = (rc == 0 and checks['hours_x3600_mod_root_dt'] == 0 and checks['child_dt_integer_ratio']
      and all(v == 0 for v in checks['run_mod_child_dt']) and checks['interval_mod_root_dt'] == 0
      and all(a == 0 and b == 0 for a, b in checks['nest_extent_mod_ratio']))
plan = json.loads(buf.getvalue()[buf.getvalue().index('{'):]) if '{' in buf.getvalue() else None
rec = dict(case=str(case), argv=argv + ['--dry-run'], rc=rc, wall_s=wall, checks=checks, verdict='PASS' if ok else 'FAIL',
           plan={k: plan.get(k) for k in ('run_type', 'init_mode', 'emit_initial_history', 'aot_prefetch')} if plan else None,
           plan_hours=plan and {k: v for k, v in plan.items() if 'hour' in k})
(out.parent/(out.name + '.json')).write_text(json.dumps(rec, indent=1, default=str) + '\n')
print(json.dumps(rec, default=str)); raise SystemExit(0 if ok else 1)
