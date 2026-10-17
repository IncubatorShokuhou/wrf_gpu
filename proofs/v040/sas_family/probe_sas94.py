"""CPU-only pristine CU_SAS component experiment; no operational qualification."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from evidence_gate import field_score, require_finite

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ("RAINCV", "PRATEC", "HBOT", "HTOP", "RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
CONFIGS = {"dt54_stepcu5": (54., 5), "dt18_stepcu1": (18., 1)}


def load_inputs():
    return dict(np.load(ROOT / "proofs/v034/scalesas/savepoints/columns_inputs.npz"))


def prepare(work):
    inp = load_inputs()
    n, kx = inp["T"].shape
    for name, (dt, stepcu) in CONFIGS.items():
        for shallow in (0, 1):
            lines = [f"{n} {kx}", f"{dt} {stepcu} 10 9000 .55 {shallow}"]
            for i in range(n):
                lines.append(f'{float(inp["XLAND"][i])} {float(inp["DX2D"][i])}')
                for key in ("T", "QV", "QC", "QI", "P", "PI", "RHO", "DZ", "U", "V", "P8W", "W"):
                    lines.append(" ".join(repr(float(x)) for x in inp[key][i]))
            (work / f"{name}_shallow{shallow}.txt").write_text("\n".join(lines) + "\n")


def parse(path):
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) % 5 == 0
    h = np.array([[float(x) for x in line[1:]] for line in lines[::5]])
    out = dict(zip(FIELDS[:4], h[:, :4].T))
    for i, key in enumerate(FIELDS[4:], 1):
        out[key] = np.array([[float(x) for x in line] for line in lines[i::5]])
    return out


def compare(work, compiled, shallow):
    import jax
    from gpuwrf.physics.cumulus_sas94 import cu_sas94_columns

    assert jax.devices()[0].platform == "cpu"
    jax.config.update("jax_enable_x64", True)
    inp = load_inputs()
    args = [inp[k] for k in ("T", "QV", "QC", "QI", "P", "PI", "RHO", "DZ", "U", "V", "W")]
    args += [inp["P8W"][:, 0], inp["XLAND"]]
    report = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "backend": "cpu", "compiled": compiled, "component": f"CU_SAS shallow={shallow}", "arms": {}}
    for cfg, (dt, stepcu) in CONFIGS.items():
        for mode in ("r4", "r8"):
            oracle = parse(work / f"{cfg}_shallow{shallow}_{mode}.out")
            fn = functools.partial(cu_sas94_columns, dt=dt, stepcu=stepcu, shallow=bool(shallow), literals=mode)
            if compiled:
                fn = jax.jit(fn)
            start = time.monotonic()
            actual = {k: np.asarray(v) for k, v in fn(*args).items()}
            arm = {"elapsed_s": time.monotonic()-start, "active_oracle": int((oracle["RAINCV"] > 0).sum()),
                   "active_port": int((actual["RAINCV"] > 0).sum()), "fields": {}}
            for key in FIELDS:
                a, b = actual[key], oracle[key]
                b=b.astype(a.dtype)
                def bound():
                    if key in ('HBOT','HTOP'):return np.zeros_like(b)
                    if mode=='r4':return np.spacing(np.maximum(np.abs(a),np.abs(b)))
                    scale=(np.max(np.abs(b),axis=1,keepdims=True) if a.ndim==2 else np.abs(b))+1e-300
                    return 1.e-9*scale
                rec=field_score(a,b,bound)
                rec['bit_equal_fraction']=rec['byte_fraction']
                if mode=='r4' and key not in ('HBOT','HTOP'):
                    rec['pass']=rec['pass'] and rec['byte_fraction'] is not None and rec['byte_fraction']>=.999
                bad=np.zeros(a.shape,bool)
                for index in rec['bad_values']:bad[tuple(index)]=True
                rec['failed_columns']=np.flatnonzero(bad.any(axis=1) if a.ndim==2 else bad).tolist()
                arm['fields'][key]=rec
            arm["pass"] = all(v["pass"] for v in arm["fields"].values()) and arm["active_oracle"] >= 10
            report["arms"][cfg+"_"+mode] = arm
            np.savez_compressed(work / f"{cfg}_{mode}_shallow{shallow}_port_{'jit' if compiled else 'eager'}.npz", **actual)
    report["pass"] = all(v["pass"] for v in report["arms"].values())
    path = work / f"comparison_shallow{shallow}_{'jit' if compiled else 'eager'}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "compare"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--shallow", type=int, choices=(0,1), default=0)
    opts = parser.parse_args()
    opts.work.mkdir(parents=True, exist_ok=True)
    if opts.action == "prepare":
        prepare(opts.work)
    else:
        raise SystemExit(compare(opts.work, opts.jit, opts.shallow))
