"""Verify tracked imports/fixtures from a clean archive, without numerical repeats."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not args.extract.exists(), "use a new lane-owned directory"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.tree, text=True).strip()
    archive = subprocess.check_output([
        "git", "archive", commit, "src", "proofs/s2_keps", "tests/test_v040_keps_operators.py",
        "tests/v025/b_core/pd_scalar_oracle.py", "tests/v025/b_core/update_scalar_oracle.py",
        "data/fixtures/thompson-tables-v1.npz",  # physics.__init__ eagerly loads this existing tracked dependency
        "data/fixtures/rrtmg-tables-v1.npz",  # coupling.__init__ -> driver -> physics_couplers
    ], cwd=args.tree)
    args.extract.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(args.extract, filter="data")
    env = dict(os.environ, PYTHONPATH=str(args.extract/"src"))
    child = """
import hashlib,json,pathlib,sys
import jax,numpy as np
from gpuwrf.physics import pbl_keps,pbl_eeps
root=pathlib.Path.cwd()
for module in (pbl_keps,pbl_eeps):
 assert pathlib.Path(module.__file__).is_relative_to(root),module.__file__
p=root/'proofs/s2_keps/operators_real32.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='c656e0edd9342af2a4bba19e6cedce47e97041804ce9f0547e0a99e6d7bb2b35'
fixture=json.loads(p.read_text())
assert len(fixture['records'])==18
assert hashlib.sha256(p.with_name('operators_oracle.f90').read_bytes()).hexdigest()==fixture['driver_sha256']
assert p.with_name('source_probe_receipt.json').exists()
assert p.with_name('manifest.yaml').exists()
full=p.with_name('full_zero.json')
assert hashlib.sha256(full.read_bytes()).hexdigest()=='546c428abf24624a959b4c86cbe83d1a3bd2ca1ef452f60dbcaa352571927fb1'
assert len(json.loads(full.read_text())['records'])==60
assert p.with_name('full_manifest.yaml').exists()
assert p.with_name('initialization_sensitivity.json').exists()
from gpuwrf.coupling import epsilon_pbl
from gpuwrf.io import pbl_epsilon_schema
assert pathlib.Path(epsilon_pbl.__file__).is_relative_to(root)
assert pathlib.Path(pbl_epsilon_schema.__file__).is_relative_to(root)
sys.path[:0]=[str(root/'proofs/s2_keps'),str(root/'tests/v025/b_core')]
import literal_real32,pd_capture,pd_oracle_check,pd_scalar_oracle,update_scalar_oracle
for module in (literal_real32,pd_capture,pd_oracle_check,pd_scalar_oracle,update_scalar_oracle):
 assert pathlib.Path(module.__file__).is_relative_to(root)
event=json.loads((root/'proofs/s2_keps/pd_qc_event.json').read_text())
arm=next(iter(event['arms'].values()))
arrays=root/arm['pd_arrays_path']
assert hashlib.sha256(arrays.read_bytes()).hexdigest()==arm['pd_arrays_sha256']
data=np.load(arrays,allow_pickle=False)
assert data['q'].shape==(44,42,42)
assert all(data[k].dtype==np.float32 and np.isfinite(data[k]).all() for k in data.files)
reference=json.loads((root/'proofs/s2_keps/pd_qc_reference.json').read_text())
assert reference['frozen_pd_gate']['passed'] and reference['wrf_reproduces_negative']
assert reference['qualification_status']=='EXPERIMENTAL_C05_PASS_WITH_DISCLOSED_POINT_INTERVAL_EXCESS'
print(json.dumps({'imports':'archive-local','records':18,'full_records':60,'fixture_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'pd_qc_arrays':{'sha256':arm['pd_arrays_sha256'],'shape':[44,42,42],'fields':len(data.files)},
 'numeric_replays':0,'gpu_cost':'UNKNOWN'}))
"""
    result = subprocess.run([sys.executable, "-c", child], cwd=args.extract, env=env,
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        args.output.write_text(json.dumps({"status": "FAIL", "source_commit": commit,
                                           "returncode": result.returncode, "stderr": result.stderr[-3000:]}, indent=2)+"\n")
        raise RuntimeError(f"archive child failed; see {args.output}")
    receipt = {"status": "PASS", "source_commit": commit, "checks": json.loads(result.stdout),
               "scope": "clean tracked archive imports/assets; no repeated numeric validation"}
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
