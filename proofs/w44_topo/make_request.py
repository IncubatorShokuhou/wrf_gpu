"""Freeze v11 tools from one commit, write FORECAST_REQUEST_v11.json, render the four launchers.

usage: python proofs/w44_topo/make_request.py <commit> <snapshot dir>
The model tree is the given detached snapshot checked out at <commit>; tools are git-show copies (read-only).
v11 = exact native shadow clock INCLUDING caller unit conversions + phase-aware mini builder; numerical GPU gate pending.
Previous v9 = v8 logic re-pinned onto origin/main 700a382f8b + topo diff (manager 08:15Z receipt-carry re-pin).
v8 = v7 + old outer cap = device cap + 30 s for the bundle consumer's post-run checks (v7 never ran).
v7 = v6 + launcher QA (cr-w44-launchers 01:06Z): E41 verdict JSON parsed, never the tool rc (E230; INCOMPLETE
with STACK PASS acceptable, FAIL/MISSING recorded without aborting); no rescan of the sealed donor (its bundle E41 +
compiled_new=0); new-cache E41 after the checker, cache kept if MISSING; own systemd scope per model child (E233).
v6 = v5 + E41 receipt-directory read fix. v5 vs v4: hours 12/3 (E187, whole root steps), real-model CPU launch smoke (stops after init), mini =
native + old warm 3 h on the real grid (the new phase's cost is its cold compile, identical to production).
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-topo')
TOOLS = ['cpu_post.py', 'device_layer.py', 'forecast_compare.py', 'forecast_driver.py', 'forecast_harness.py',
         'gpu_dispatch.py', 'gpu_probe.py']
CASE = Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227')
INPUTS = ['namelist.input', 'wrfbdy_d01', 'wrfinput_d01', 'wrfinput_d02', 'wrfinput_d03',
          'wrflowinp_d01', 'wrflowinp_d02', 'wrflowinp_d03']
DEPENDENCIES = [LANE / 'night_tools_111bc8fb6b10/night_driver.py',
                Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/sealed/BUNDLE_MANIFEST.json'),
                Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/e41/wave_gate.json'),
                Path('<USER_HOME>/miniconda3/bin/python'),
                LANE / 'TP19_fixture/manifest.json', LANE / 'TP19_fixture/fixture.npz',
                LANE / 'cpu_gate_fixture/MANIFEST.json',
                *sorted((LANE / 'cpu_gate_fixture').glob('jit__lambda-*-cache'))]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main():
    commit = subprocess.check_output(['git', 'rev-parse', sys.argv[1]], cwd=REPO, text=True).strip()
    TREE = Path(sys.argv[2]).resolve()
    assert TREE.parent == LANE and (TREE / 'data/wrf_pristine').resolve() == Path('<USER_HOME>/src/wrf_pristine')
    assert subprocess.check_output(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', str(TREE), 'status', '--porcelain', '--untracked-files=no'])
    tools = LANE / f'forecast_tools_v11_{commit[:12]}'
    tools.mkdir()
    for name in TOOLS:
        (tools / name).write_bytes(subprocess.check_output(['git', 'show', f'{commit}:proofs/w44_topo/{name}'], cwd=REPO))
        os.chmod(tools / name, 0o444)
    os.chmod(tools, 0o555)
    production = dict(input_dir=str(CASE), native_manifest=str(LANE / 'TP19_fixture/manifest.json'), hours=12,
                      report_start=8, phases=['native', 'old', 'new', 'checker'],
                      caps=dict(native=120, old=650, old_post=30, new=1400, e41=60, checker=40, e41_new=400),
                      input_hashes={n: sha(CASE / n) for n in INPUTS})
    mini = dict(production, hours=3, report_start=0, phases=['native', 'old'],
                caps=dict(native=50, old=120, old_post=30, e41=55, checker=0, e41_new=0))
    spec = dict(schema=11, created_utc=datetime.now(timezone.utc).isoformat(), tree=str(TREE), pin=commit,
                tools=str(tools), tools_commit=commit, old_bundle='<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3',
                cpu_gate_fixture=str(LANE / 'cpu_gate_fixture'),
                smoke_caps=dict(native=900, old=3000, old_post=120, new=3000, e41=300, checker=120, e41_new=900),
                scope_memory=dict(gpu='16G', smoke='12G'),
                tool_hashes={n: sha(tools / n) for n in TOOLS},
                dependency_hashes={str(p): sha(p) for p in DEPENDENCIES},
                profiles=dict(production=production, mini=mini),
                differences=dict(
                    run_length='production 12 h (800 root steps at dt 54; report 08-12Z), mini 3 h (200 steps); both whole root steps (E187)',
                    mini='same builder/env/case/bundle; phases native+E41 and old v0.4.3 warm 3 h only (donor E41 = '
                         'sealed bundle receipt, compiled_new must be 0). The new phase is not miniaturisable: its cost '
                         'is the cold 3-nest compile, identical to production',
                    launch_smoke='CPU: real model hook + CLI through case load, cold init and the hours/root-dt check, '
                                 'stopped at the first post-init call; only device admission differs; model children '
                                 'in own 12G scope under the caller ram slot; native oracle on CPU interpret; native E41 '
                                 'on retained CUDA JIT fixture, new E41 on the real AOT-bearing v0.4.3 bundle cache'))
    request = LANE / 'FORECAST_REQUEST_v11.json'
    assert not request.exists()
    request.write_text(json.dumps(spec, indent=2) + '\n')
    os.chmod(request, 0o444)
    template = (REPO / 'proofs/w44_topo/night_forecast.sh.in').read_text()
    rendered = {}
    for name, cap, variant in (('night_forecast_v11.sh', 2430, ''), ('cuda_mini_v11.sh', 300, '--cuda-mini'),
                               ('launch_smoke_v11.sh', 7230, '--launch-smoke'),
                               ('launch_smoke_mini_v11.sh', 7230, '--cuda-mini --launch-smoke')):
        text = (template.replace('@TOPO_TOOLS@', str(tools)).replace('@REQUEST_PATH@', str(request))
                .replace('@DRIVER_SHA@', spec['tool_hashes']['forecast_driver.py'])
                .replace('@REQUEST_SHA@', sha(request)).replace('@CAP_SECONDS@', str(cap))
                .replace('@VARIANT_ARGS@', variant))
        assert '@' not in text.replace('"$@"', '')
        path = LANE / name
        assert not path.exists()
        path.write_text(text)
        os.chmod(path, 0o555)
        rendered[name] = sha(path)
    print(json.dumps(dict(pin=commit, tools=str(tools), request=str(request), request_sha256=sha(request),
                          launchers=rendered), indent=2))


if __name__ == '__main__':
    main()
