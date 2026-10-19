#!/usr/bin/env python3
"""w42-bundle042 CPU RC-suite runner.

Method = the v0.4.1 full-suite method (w41-knownred): ONE FRESH pytest process per test file, so JIT caches,
env flags and memory never leak across files (E161/E183), RAM tier per file from the v0.4.1 measurements.
Additions: QUIET-aware (no new file while /tmp/wrf_gpu2_quiet exists; a running file's own process group is
SIGSTOPped and SIGCONTed, and its timeout counts RUNNING time only) and a shared claim queue (O_EXCL claim
files, never mkdir: E207) so several workers (one RAM scope + one core each) drain one LPT-ordered plan.

  rc_suite.py collect --tree T --out O                      whole-tree collect-only (rc captured first), node list
  rc_suite.py plan    --tree T --out O [--baseline J] [--file-list F] PATH...   files + tiers + timeouts
  rc_suite.py worker  --tree T --out O --name A --tiers 6G[,12G,20G]   (run inside with_ram_slot + taskset)
  rc_suite.py report  --out O
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

QUIET = Path('/tmp/wrf_gpu2_quiet')
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w42-bundle042')
FC = '<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran'
TIER_RANK = {'6G': 0, '12G': 1, '20G': 2}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def key(path):
    return path.replace('/', '__').removesuffix('.py')


def head_of(tree):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()


def src_tree(tree):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD:src'], cwd=tree, text=True).strip()


def base_env(tree, tmp):
    env = dict(os.environ)
    for k in ('GPUWRF_GPU_LOCK_HELD', 'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR', 'GPUWRF_JAX_CACHE_DIR'):
        env.pop(k, None)
    env.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', GPUWRF_JAX_CACHE='0', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(Path(tree) / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TMPDIR=str(tmp), XDG_CACHE_HOME=str(tmp / 'xdg'),
               TRITON_CACHE_DIR=str(tmp / 'triton'), FC=FC, GF=FC,
               GPUWRF_WRF_ROOT=str(Path(tree) / 'data/wrf_pristine/WRF'), BENCH_SRC_REPO=str(tree))
    return env


def pytest_cmd(tree, basetemp, xml, files, extra=()):
    return [sys.executable, '-m', 'pytest', '-c', str(Path(tree) / 'pyproject.toml'), '--rootdir', str(tree),
            '-p', 'no:cacheprovider', '--basetemp', str(basetemp), '--junitxml', str(xml), '-q', '-ra',
            *extra, *files]


def wait_quiet(log):
    said = False
    while QUIET.exists():
        if not said:
            log(f'QUIET wait ({QUIET.read_text().strip()[:80] if QUIET.exists() else ""})')
            said = True
        time.sleep(5)


def mem_available_gib():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1048576
    return 0.0


def group_cpu_ticks(pgid):
    """utime+stime of every live process in the process group (E105/E119 stall detection)."""
    tot = 0
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            f = open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()
        except OSError:
            continue
        if int(f[2]) == pgid:  # field 5 = pgrp
            tot += int(f[11]) + int(f[12])  # fields 14/15 = utime/stime
    return tot


def run_paused(cmd, cwd, env, logf, timeout_active, log, stall_s=900):
    """Run cmd in its own session; SIGSTOP its group while QUIET exists; timeout on running time only;
    kill a group whose CPU time does not advance for stall_s seconds of running time (E105 futex freeze)."""
    with open(logf, 'w') as out:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.monotonic()
    active = 0.0
    last = t0
    paused = False
    pauses = []
    killed = None
    ticks, ticks_at, checked = -1, 0.0, -1e9
    while True:
        rc = p.poll()
        now = time.monotonic()
        if not paused:
            active += now - last
        last = now
        if rc is not None:
            break
        if not paused and active - checked >= 30:
            checked = active
            t = group_cpu_ticks(p.pid)
            if t != ticks:
                ticks, ticks_at = t, active
            elif active - ticks_at > stall_s and killed is None:
                log(f'  STALL: no CPU progress for {active - ticks_at:.0f} s running time: SIGINT pgid {p.pid}')
                killed = 'STALL'
                os.killpg(p.pid, signal.SIGINT)
                kill_at = active + 30
        if QUIET.exists() and not paused:
            os.killpg(p.pid, signal.SIGSTOP)
            paused = True
            pauses.append([utc(), None])
            log(f'  SIGSTOP pgid {p.pid} (QUIET)')
        elif paused and not QUIET.exists():
            os.killpg(p.pid, signal.SIGCONT)
            paused = False
            pauses[-1][1] = utc()
            log(f'  SIGCONT pgid {p.pid}')
        if not paused and active > timeout_active and killed is None:
            killed = 'INT'
            os.killpg(p.pid, signal.SIGINT)
            kill_at = active + 30
        elif killed in ('INT', 'STALL') and not paused and active > kill_at:
            killed += '+KILL'
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(2)
    return dict(rc=rc, wall_s=round(time.monotonic() - t0, 3), active_s=round(active, 3), pauses=pauses,
                timeout_active_s=timeout_active, killed=killed)


def parse_xml(xml):
    """Node outcomes from JUnit XML: PASS/FAIL/ERROR/SKIP/XFAIL (XPASS needs the -ra text, see parse_ra)."""
    nodes = {}
    if not Path(xml).exists():
        return nodes
    root = ET.parse(xml).getroot()
    for tc in root.iter('testcase'):
        cls = tc.get('classname', '')
        name = tc.get('name', '')
        nid = f'{cls}::{name}'
        outcome, msg = 'PASS', ''
        for ch in tc:
            if ch.tag == 'failure':
                outcome, msg = 'FAIL', (ch.get('message') or '')
            elif ch.tag == 'error':
                outcome, msg = 'ERROR', (ch.get('message') or '')
            elif ch.tag == 'skipped':
                if (ch.get('type') or '') == 'pytest.xfail':
                    outcome, msg = 'XFAIL', (ch.get('message') or '')
                else:
                    outcome, msg = 'SKIP', (ch.get('message') or '')
        if nid in nodes and nodes[nid][0] in ('FAIL', 'ERROR'):
            continue  # keep a call failure over a teardown record of the same node
        nodes[nid] = (outcome, msg[:300])
    return nodes


RA = re.compile(r'^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)(?:\s+-\s+(.*))?$')


def parse_ra(logf):
    """Short-summary lines (-ra): FAILED/ERROR/XPASS/XFAIL node ids with reasons, and the final summary line."""
    lines = Path(logf).read_text(errors='replace').splitlines() if Path(logf).exists() else []
    out = {'FAILED': [], 'ERROR': [], 'XPASS': [], 'XFAIL': []}
    for ln in lines:
        m = RA.match(ln)
        if m and m.group(1) in out:
            out[m.group(1)].append([m.group(2), (m.group(3) or '')[:300]])
    summary = next((ln for ln in reversed(lines) if re.search(r'(passed|failed|error|skipped|no tests ran)', ln)), '')
    return out, summary.strip('= ').strip()


def cmd_collect(a):
    tree = Path(a.tree)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wait_quiet(print)
    tmp = Path(a.scratch) / 'collect_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    env = base_env(tree, tmp)
    cmd = [sys.executable, '-m', 'pytest', '-c', str(tree / 'pyproject.toml'), '--rootdir', str(tree), '-p',
           'no:cacheprovider', '--collect-only', '-q', '-rs', *a.paths]
    t0 = time.monotonic()
    p = subprocess.run(['/usr/bin/time', '-v', '-o', str(out / 'collect.time'), *cmd], cwd=tree, env=env,
                       capture_output=True, text=True)
    rc = p.returncode  # captured before anything else is printed
    (out / 'collect.log').write_text(p.stdout + p.stderr)
    nodes = [ln for ln in p.stdout.splitlines() if '::' in ln and not ln.startswith(('SKIPPED', 'ERROR'))]
    errors = [ln for ln in p.stdout.splitlines() if ln.startswith('ERROR')]
    skips = [ln for ln in p.stdout.splitlines() if ln.startswith('SKIPPED')]
    last = next((ln for ln in reversed(p.stdout.splitlines()) if ln.strip()), '')
    rec = dict(utc=utc(), tree=str(tree), head=head_of(tree), src_tree=src_tree(tree), rc=rc, paths=a.paths,
               nodes=len(nodes), errors=errors, module_skips=skips, summary=last,
               wall_s=round(time.monotonic() - t0, 2))
    (out / 'collect_nodes.txt').write_text('\n'.join(nodes) + '\n')
    (out / 'collect.json').write_text(json.dumps(rec, indent=2) + '\n')
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: rec[k] for k in ('rc', 'nodes', 'summary', 'wall_s')}), 'errors', len(errors))
    return 0 if rc == 0 else 1


def cmd_plan(a):
    tree = Path(a.tree)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = {}
    if a.baseline:
        for r in json.load(open(a.baseline)):
            base[r['file']] = r
    nodes = {}
    if a.nodes:
        for ln in open(a.nodes):
            ln = ln.strip()
            if '::' in ln:
                f = ln.split('::')[0]
                nodes[f] = nodes.get(f, 0) + 1
    files = set()
    for pth in a.paths:
        p = tree / pth
        if p.is_file():
            files.add(pth)
        else:
            for f in p.rglob('*.py'):
                if f.name.startswith('test_') or f.name.endswith('_test.py'):
                    files.add(str(f.relative_to(tree)))
    if a.file_list:
        files |= {ln.strip() for ln in open(a.file_list) if ln.strip()}
    plan = []
    for f in sorted(files):
        b = base.get(f)
        wall = (b or {}).get('wall_s') or None
        tier = (b or {}).get('ram') or '6G'
        if tier not in TIER_RANK:
            tier = '20G'
        timeout = 1800 if wall is None else int(min(3600, max(600, 6 * wall)))
        plan.append(dict(file=f, tier=tier, base_wall_s=wall, base_counts=(b or {}).get('counts'),
                         expected_nodes=nodes.get(f), timeout_active_s=timeout))
    plan.sort(key=lambda r: (-(r['base_wall_s'] or 120.0), r['file']))  # LPT; unknown files at 120 s
    rec = dict(utc=utc(), tree=str(tree), head=head_of(tree), src_tree=src_tree(tree), paths=a.paths,
               file_list=a.file_list, baseline=a.baseline, n=len(plan),
               tiers={t: sum(1 for r in plan if r['tier'] == t) for t in TIER_RANK},
               base_wall_s=round(sum(r['base_wall_s'] or 120.0 for r in plan)), files=plan)
    (out / 'plan.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(f"plan {rec['n']} files tiers {rec['tiers']} baseline wall {rec['base_wall_s']} s head {rec['head'][:10]}")
    return 0


def cmd_worker(a):
    tree = Path(a.tree)
    out = Path(a.out)
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['head'] == head_of(tree), 'tree moved since plan'
    assert os.environ.get('JAX_PLATFORMS') == 'cpu', 'set JAX_PLATFORMS=cpu on the worker line (E132)'
    tiers = a.tiers.split(',')
    order = sorted(plan['files'], key=lambda r: -TIER_RANK[r['tier']]) if len(tiers) > 1 else plan['files']
    wlog = open(out / f'worker_{a.name}.log', 'a')

    def log(msg):
        line = f'{utc()} [{a.name}] {msg}'
        print(line, flush=True)
        wlog.write(line + '\n')
        wlog.flush()

    log(f'START pid {os.getpid()} tiers {tiers} affinity {sorted(os.sched_getaffinity(0))} '
        f'cgroup {Path("/proc/self/cgroup").read_text().strip().split("/")[-1]}')
    deadline = datetime.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else None
    done = 0
    for job in order:
        if job['tier'] not in tiers:
            continue
        d = out / key(job['file'])
        d.mkdir(exist_ok=True)
        if (d / 'result.json').exists():
            continue
        try:
            fd = os.open(d / 'claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f'{a.name} {os.getpid()} {utc()}\n'.encode())
        os.close(fd)
        wait_quiet(log)
        if deadline and time.time() + 60 > deadline:
            os.unlink(d / 'claim')
            log('deadline reached; claim released, stopping')
            break
        if job['tier'] == '20G':
            while mem_available_gib() < a.min_avail_20g:
                log(f'  waiting MemAvailable {mem_available_gib():.1f} GiB < {a.min_avail_20g} for 20G file')
                time.sleep(30)
                wait_quiet(log)
        tmp = Path(a.scratch) / f'tmp_{a.name}'
        bt = LANE / 'pytest' / a.name
        for pth in (tmp, bt):
            shutil.rmtree(pth, ignore_errors=True)
        tmp.mkdir(parents=True)
        bt.parent.mkdir(parents=True, exist_ok=True)   # pytest creates --basetemp itself but not its parent (E202)
        env = base_env(tree, tmp)
        cmd = ['/usr/bin/time', '-v', '-o', str(d / 'time.txt'),
               *pytest_cmd(tree, bt, d / 'pytest.xml', [job['file']], a.extra.split() if a.extra else ())]
        log(f"BEGIN {job['file']} tier {job['tier']} base {job['base_wall_s']} timeout {job['timeout_active_s']}")
        res = run_paused(cmd, tree, env, d / 'pytest.log', job['timeout_active_s'], log)
        nodes = parse_xml(d / 'pytest.xml')
        ra, summary = parse_ra(d / 'pytest.log')
        xpass = {n for n, _ in ra['XPASS']}
        counts = {}
        for nid, (o, _) in nodes.items():
            counts[o] = counts.get(o, 0) + 1
        maxrss = None
        if (d / 'time.txt').exists():
            m = re.search(r'Maximum resident set size \(kbytes\): (\d+)', (d / 'time.txt').read_text())
            maxrss = int(m.group(1)) // 1024 if m else None
        result = dict(job, worker=a.name, head=plan['head'], **res, summary=summary, counts=counts,
                      xpass=sorted(xpass), failed=ra['FAILED'], errors=ra['ERROR'],
                      fails_xml={n: m for n, (o, m) in nodes.items() if o in ('FAIL', 'ERROR')},
                      maxrss_mib=maxrss, finished=utc())
        (d / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        log(f"END {job['file']} rc {res['rc']} {summary} active {res['active_s']:.0f}s rss {maxrss}MiB")
        for pth in (tmp, bt):
            shutil.rmtree(pth, ignore_errors=True)
        done += 1
    log(f'DONE {done} files')
    return 0


def cmd_report(a):
    out = Path(a.out)
    plan = json.loads((out / 'plan.json').read_text())
    rows, missing = [], []
    for job in plan['files']:
        rp = out / key(job['file']) / 'result.json'
        if not rp.exists():
            missing.append(job['file'])
            continue
        rows.append(json.loads(rp.read_text()))
    per_dir = {}
    tot = {}
    fails = []
    for r in rows:
        parts = r['file'].split('/')
        dk = '/'.join(parts[:3]) if parts[1] in ('v025', 'v040') and len(parts) > 3 else '/'.join(parts[:2])
        c = dict(r['counts'])
        if r['xpass']:
            c['XPASS'] = len(r['xpass'])
            c['PASS'] = c.get('PASS', 0) - len(r['xpass'])
        if not c and r['rc'] not in (0, 5):
            c['ERROR'] = 1
        for k, v in c.items():
            per_dir.setdefault(dk, {})[k] = per_dir.setdefault(dk, {}).get(k, 0) + v
            tot[k] = tot.get(k, 0) + v
        if r['rc'] not in (0, 5) or r['fails_xml'] or r['failed'] or r['errors']:
            ra_ids = [n for n, _ in r['failed'] + r['errors']]   # real pytest node ids from the -ra summary
            fails.append(dict(file=r['file'], rc=r['rc'], summary=r['summary'], killed=r['killed'],
                              nodes=sorted(set(ra_ids)) or sorted(r['fails_xml']),   # JUnit 'classname::name' only as fallback
                              msgs=r['fails_xml']))
    rep = dict(utc=utc(), head=plan['head'], src_tree=plan['src_tree'], files_planned=plan['n'],
               files_done=len(rows), missing=missing, totals=tot, per_dir=per_dir, failing_files=fails,
               active_s=round(sum(r['active_s'] for r in rows)), max_rss_mib=max((r['maxrss_mib'] or 0) for r in rows) if rows else 0,
               expected_vs_observed=[(r['file'], r['expected_nodes'], sum(r['counts'].values())) for r in rows
                                     if r['expected_nodes'] is not None and r['expected_nodes'] != sum(r['counts'].values())])
    (out / 'report.json').write_text(json.dumps(rep, indent=2) + '\n')
    print(json.dumps(dict(files=f"{len(rows)}/{plan['n']}", totals=tot, failing_files=len(fails),
                          missing=len(missing), active_s=rep['active_s'])))
    for f in fails:
        print('FAILFILE', f['file'], f['rc'], f['summary'][:120])
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for n in ('collect', 'plan', 'worker', 'report'):
        s = sub.add_parser(n)
        s.add_argument('--out', required=True)
        if n != 'report':
            s.add_argument('--tree', required=True)
            s.add_argument('--scratch', default='<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042')
        if n in ('collect', 'plan'):
            s.add_argument('paths', nargs='*')
        if n == 'plan':
            s.add_argument('--baseline')
            s.add_argument('--nodes')
            s.add_argument('--file-list')
        if n == 'worker':
            s.add_argument('--name', required=True)
            s.add_argument('--tiers', default='6G')
            s.add_argument('--deadline')
            s.add_argument('--extra', default='')
            s.add_argument('--min-avail-20g', type=float, default=34.0)
    a = ap.parse_args()
    return {'collect': cmd_collect, 'plan': cmd_plan, 'worker': cmd_worker, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
