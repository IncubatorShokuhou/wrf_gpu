"""Clean tracked archive imports and original-FFTPACK fixture availability."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tree",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    tree=args.tree.resolve()
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=tree,text=True).strip()
    with tempfile.TemporaryDirectory(prefix="clean-archive-",dir=args.output.parent) as scratch:
        root=Path(scratch)
        archive=root/"source.tar"
        with archive.open("wb") as f:
            subprocess.run(["git","archive",commit,"src","proofs/u2_stoch","data/fixtures",
                "tests/fixtures/u2_stoch_real4.npz","tests/fixtures/u2_stoch_real4.json"],
                cwd=tree,stdout=f,check=True,timeout=60)
        with tarfile.open(archive) as f:
            f.extractall(root,filter="data")
        archive.unlink()
        (root/"data").mkdir(exist_ok=True)
        (root/"data/wrf_pristine").symlink_to("<USER_HOME>/src/wrf_pristine")
        code='''import hashlib,json,sys
from pathlib import Path
import jax,numpy as np
from gpuwrf.physics import stochastic
root=Path.cwd()
assert Path(stochastic.__file__).resolve().is_relative_to(root.resolve())
fixture=root/"tests/fixtures/u2_stoch_real4.npz"
manifest=json.loads(fixture.with_suffix(".json").read_text())
assert hashlib.sha256(fixture.read_bytes()).hexdigest()==manifest["fixture_sha256"]
with np.load(fixture) as data:
    assert len(data.files)==106
    result=stochastic.spectral_field(*[jax.numpy.asarray(data[k]) for k in
        ("sine_3","cosine_3")],component="T",dx=9000.,dy=9000.,
        **{k:jax.numpy.asarray(data[k]) for k in ("vertical_cosine","vertical_sine","vertical_amplitude")})
    truth=data["field_3_1_T"]
    np.testing.assert_allclose(result,truth,rtol=4e-6,atol=64*np.finfo(np.float32).eps*np.max(np.abs(truth)))
from gpuwrf.perturb import inputs
assert Path(inputs.__file__).resolve().is_relative_to(root.resolve())
print(json.dumps({"arrays":106,"sha256":manifest["fixture_sha256"],"originalFFTgate":"PASS","imports":"archive only"}))
'''
        result=subprocess.run([sys.executable,"-c",code],cwd=root,
            env={**os.environ,"PYTHONPATH":str(root/"src"),"JAX_PLATFORMS":"cpu","GPUWRF_JAX_CACHE":"0"},
            text=True,capture_output=True,timeout=120)
        assert result.returncode==0,result.stderr
        receipt=dict(commit=commit,backend="cpu",GPU_execution=False,archive=json.loads(result.stdout),
            cache="disabled",scratch="temporary archive removed immediately")
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
