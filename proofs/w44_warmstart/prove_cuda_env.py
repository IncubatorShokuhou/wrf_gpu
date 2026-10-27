"""CPU proof of the exact production env builders and child Python commands.

Only JAX_PLATFORMS=cpu and an explicit CPU-only smoke selector differ from
the product child env. No GPU lock, NVML wrapper or mocked GPU device is used.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    assert os.environ['JAX_PLATFORMS']=='cpu'
    component_dir,forecast_dir,root=map(Path,sys.argv[1:4]);root.mkdir(exist_ok=False,parents=True)
    sys.path[:0]=[str(component_dir),str(forecast_dir)]
    from launch_preflight import cuda_environment,CPU_PREFLIGHT
    component=module('tested_component',component_dir/'night_component.py')
    forecast=module('tested_forecast',forecast_dir/'night_forecast.py')
    warm=module('tested_warm',forecast_dir/'night_warm_cache.py')
    snap=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/WS19_snapshot')
    oracle=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/WS11_mynn')
    old=cuda_environment({'JAX_PLATFORMS':'gpu','JAX_PLATFORM_NAME':'rocm',CPU_PREFLIGHT:'1'})
    assert old['JAX_PLATFORMS']=='cuda'and old['CUDA_VISIBLE_DEVICES']=='0'
    assert'JAX_PLATFORM_NAME'not in old and CPU_PREFLIGHT not in old
    rows=[]
    for name in ('component','forecast','warm'):
        run=root/name;run.mkdir();(run/'cache/jax').mkdir(parents=True)
        for folder in ('cache/triton','cache/cuda','tmp'):(run/folder).mkdir()
        if name=='component':
            production=component.child_environment(snap,run)
            seal=json.loads(component.SEAL.read_text())
            command=component.child_command(seal,run)
            report_path=run/'probe/PREFLIGHT.json'
        elif name=='forecast':
            production,configuration=forecast.child_environment(snap,run)
            assert configuration['JAX_PLATFORMS']=='cuda'
            seal=json.loads(forecast.SEAL.read_text())
            command=forecast.child_command(seal,seal['case'],run/'cold','cold')
            report_path=run/'cold/forecast/PREFLIGHT.json'
        else:
            production=warm.child_environment(configuration)
            command=warm.child_command(seal,seal['case'],run/'warm','warm')
            report_path=run/'warm/forecast/PREFLIGHT.json'
        assert production['JAX_PLATFORMS']=='cuda'and production['CUDA_VISIBLE_DEVICES']=='0'
        assert'JAX_PLATFORM_NAME'not in production and CPU_PREFLIGHT not in production
        dry=dict(production,JAX_PLATFORMS='cpu',**{CPU_PREFLIGHT:'1'})
        differences={k for k in production.keys()|dry.keys()if production.get(k)!=dry.get(k)}
        assert differences=={'JAX_PLATFORMS',CPU_PREFLIGHT},differences
        started=time.monotonic()
        child=subprocess.run(command,env=dry,text=True,capture_output=True,timeout=90)
        (run/'stdout.log').write_text(child.stdout);(run/'stderr.log').write_text(child.stderr)
        assert child.returncode==0,(name,child.returncode,child.stderr[-3000:])
        report=json.loads(report_path.read_text())
        assert report['platform']=='cpu'and report['model_body_reached']
        assert report['verdict']=='CPU_ENVIRONMENT_SMOKE_PASS'
        assert sum(r['columns']for r in report['records'])==4
        assert not report_path.with_name('result.json').exists(),'CPU smoke cannot publish GPU-oracle PASS'
        rows.append({'launcher':name,'production_requested_platform':production['JAX_PLATFORMS'],
            'production_visible_devices':production['CUDA_VISIBLE_DEVICES'],'CPU_command':command,
            'only_env_changes':sorted(differences),'rc':child.returncode,'wall_s':time.monotonic()-started,
            'model_report':str(report_path),'model_body_reached':True,'real_device':'cpu'})
    import jax
    from jax._src import xla_bridge
    aliases={name:xla_bridge.expand_platform_alias(name)for name in ('gpu','cuda','cpu')}
    assert aliases['gpu']==['cuda','rocm']and aliases['cuda']==['cuda'],aliases
    assert jax.devices()[0].platform=='cpu'
    report={'passed':True,'checks':rows,'installed_JAX_platform_aliases':aliases,
            'GPU_executed':False,'scope':'environment/dispatch smoke on real-column mini grids, not GPU or physics fidelity'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
