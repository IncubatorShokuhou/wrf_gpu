"""Deletion-sensitive oracle control: remove WRF's energy normalization in memory."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import types


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    source=(root/'src/gpuwrf/physics/wind_fitch.py').read_text()
    old='        ec = jnp.where((power1 == 0.) | (power2 == 0.), 1., power1/jnp.where(power2 == 0., 1., power2))'
    assert source.count(old)==1
    changed=source.replace(old,'        ec = jnp.float32(1.)')
    mutant=types.ModuleType('fitch_energy_mutant');exec(compile(changed,'<Fitch energy-normalization mutant>','exec'),mutant.__dict__)
    spec=importlib.util.spec_from_file_location('windfarm_tests',root/'tests/test_u2_windfarm.py')
    test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
    test.dragforce=mutant.dragforce
    result={'mutation':'omit WRF power1/power2 energy normalization','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'mutant_sha256':hashlib.sha256(changed.encode()).hexdigest(),'product_files_changed':False}
    try:test.test_pristine_real32_dragforce_all_regimes()
    except AssertionError as exc:result.update(killed=True,failure=str(exc)[:1200])
    else:result['killed']=False
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['killed']


if __name__=='__main__':main()
