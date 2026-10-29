"""v45-rc v0.4.5: validate the frozen PIN manifest and the trees/bundles it names; write a bash env file.

usage: python3 -I manifest_check.py MANIFEST ROLE MODE OUT_ENV OUT_JSON
  ROLE: short | rw3 | mini_short | mini_rw3     (which bundle the entry uses and what state it must be in)
  MODE: gpu | smoke                              (gpu refuses a standin manifest, a version != 0.4.5 and a moved main:src)
Stdlib only (no JAX/gpuwrf import); reads git state through `git` and file contents through SHA-256 (no st_dev/st_ino).
Exit 0 = OK (OUT_ENV + OUT_JSON written); 4 = refused (reason on stdout and in OUT_JSON).
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

man_path, role, mode, out_env, out_json = sys.argv[1:6]
assert role in ("short", "rw3", "mini_short", "mini_rw3", "prime_swiss", "prime_prod", "prime_wn3", "warm") and mode in ("gpu", "smoke"), sys.argv
res = dict(manifest=man_path, role=role, mode=mode, checks={}, info={})


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



def paid_prime(record, pin, mode):
    """A completed paid prime must bind the same source, real backend and immutable output receipts."""
    record = Path(record); row = json.loads(record.read_text()); arm = Path(row['arm'])
    assert row['pin'] == pin and row['mode'] == mode
    assert (arm/'rc.txt').read_text().strip() == '0'
    assert (arm/'source_head.txt').read_text().strip() == pin
    timer = json.loads((arm/'timer.json').read_text()); assert timer['teardown']['all_reaped']
    if mode == 'smoke' and row.get('CPU_model_gate') == 'PLAN_ONLY':
        assert row['admission']['scope'] == 'PLAN_ONLY'
        return dict(mode=mode, CPU_model_gate='PLAN_ONLY', arm=str(arm))
    proof = arm/'proof/nested_pipeline_run.json'; listing = arm/'wrfout_sha.txt'
    assert Path(row['proof']) == proof and Path(row['sha_list']) == listing
    assert sha(proof) == row['proof_sha256'] and sha(listing) == row['sha_list_sha256']
    assert row['backend']['platform'] == ('gpu' if mode == 'gpu' else 'cpu')
    measured = json.loads((arm/'device_measured.json').read_text())
    assert measured == row['backend']
    evidence = json.loads(proof.read_text()); assert evidence['all_domains_finite'] is True
    receipt = json.loads((arm/'RECEIPT.json').read_text())
    count = {'A1_swiss':4, 'A2_prod':6, 'A3_wn3_0227_3h':12}[arm.name]
    assert receipt['n_wrfout'] == count if mode == 'gpu' else receipt['n_wrfout'] > 0
    assert len([line for line in listing.read_text().splitlines() if line.strip()]) == receipt['n_wrfout']
    return dict(mode=mode, arm=str(arm), proof_sha256=sha(proof), sha_list_sha256=sha(listing), n_wrfout=receipt['n_wrfout'])


def git(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True, text=True).stdout.strip()


def refuse(msg):
    res["refused"] = msg
    Path(out_json).write_text(json.dumps(res, indent=1) + "\n")
    print("MANIFEST_REFUSED:", msg)
    sys.exit(4)


def check(name, ok, detail=None):
    res["checks"][name] = dict(ok=bool(ok), detail=detail)
    if not ok:
        refuse(f"{name}: {detail}")


try:
    man = json.loads(Path(man_path).read_text())
except (OSError, ValueError) as exc:
    refuse(f"manifest unreadable: {exc}")
res["manifest_sha256"] = sha(man_path)
check("schema", man.get("schema") == "v45-rc.pin-manifest.v1", man.get("schema"))
if mode == "gpu":
    check("not_standin", man["standin"] is False, "standin manifests are CPU-smoke only")
    check("version", man["version"] == "0.4.5" and man["pyproject_version"] == "0.4.5",
          f"{man['version']}/{man['pyproject_version']}")
tools = Path(man["tools_dir"])
check("tools_unchanged", sha(tools / "SHA256SUMS") == man["tools_sha256sums_sha256"],
      "tools_v6/SHA256SUMS changed after prepare: re-run prepare on a fresh pin manifest")
pin, tree = man["pin"], man["src_tree"]
for label, d in sorted({("snapshot", man["snapshot"]), ("prod_src", man["bundles"]["prod"]["src"]),
                         ("mini_src", man["bundles"]["mini"]["src"])}):
    try:
        head, htree = git(d, "rev-parse", "HEAD"), git(d, "rev-parse", "HEAD:src")
        # real pins: the whole detached worktree is clean; a standin (existing shared tree) only needs a clean src/
        scope = ["--", "src", "scripts"]  # one pinned slim archive, all product source/scripts tracked and clean
        dirty = git(d, "status", "--porcelain", "--untracked-files=all", *scope)
        detached = subprocess.run(["git", "-C", d, "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode != 0
    except subprocess.CalledProcessError as exc:
        refuse(f"{label} git: {exc}")
    check(f"{label}_at_pin", head == pin and htree == tree, f"{d}: HEAD {head} src {htree}")
    check(f"{label}_clean", dirty == "" and (detached or man["standin"]), f"{d}: dirty={dirty[:200]!r} detached={detached}")
    check(f"{label}_pristine", Path(d, "data/wrf_pristine/WRF/run/MPTABLE.TBL").is_file(), f"{d}/data/wrf_pristine")
main_tree = git(man["repo"], "rev-parse", "main:src")
res["info"]["main_src_tree_now"] = main_tree
diverged = main_tree != tree
if mode == "gpu" and not man["allow_src_diverge"]:
    check("source_freeze", not diverged, f"main:src {main_tree} != pinned src {tree} (manager: re-pin, or prepare --allow-src-diverge)")
res["info"]["src_diverged_from_main"] = diverged
cdir = Path(man["cases"]["dir"])
check("cases_sums_file", sha(man["cases"]["sums_file"]) == man["cases"]["sums_sha256"], man["cases"]["sums_file"])
bad = []
for line in Path(man["cases"]["sums_file"]).read_text().splitlines():
    if line.strip():
        digest, rel = line.split(None, 1)
        if not (cdir / rel).is_file() or sha(cdir / rel) != digest:
            bad.append(rel)
check("cases_content", not bad, bad[:5])
for key, files in man["input_sha256"].items():
    for f, digest in files.items():
        check(f"input_{key}_{f}", sha(Path(man["inputs"][key]) / f) == digest, f"{man['inputs'][key]}/{f}")
old = Path(man["old_bundle"]["dir"])
check("old_bundle_manifest", sha(old / "sealed/BUNDLE_MANIFEST.json") == man["old_bundle"]["manifest_sha256"], str(old))
check("old_bundle_tools", (old / "tools/bundle_consumer.sh").is_file() and (old / "tools/compile_bundle.py").is_file(), str(old))
bkey = "mini" if role.startswith("mini") else "prod"
b = Path(man["bundles"][bkey]["dir"])
bsrc = Path(man["bundles"][bkey]["src"])
expect_env = (f"# v0.4.5 sealed-bundle environment (v45-rc prepare, pin {pin}): source this file unchanged in priming AND in every\n"
              "# consumer arm (release defaults: set no other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT)\n"
              f"export PYTHONPATH={bsrc}/src\n"
              f"export GPUWRF_JAX_CACHE_DIR={b}/jax JAX_COMPILATION_CACHE_DIR={b}/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR={b}/autotune\n"
              f"export TRITON_CACHE_DIR={b}/triton CUDA_CACHE_PATH={b}/cuda\n")
check("bundle_env", (b / "bundle.env").read_text() == expect_env, str(b / "bundle.env"))
sums = (b / "tools/SHA256SUMS").read_text().split("\n")
check("bundle_tools", all(sha(b / "tools" / l.split()[1]) == l.split()[0] for l in sums if l.strip()), str(b / "tools"))
for f in ("compile_bundle.py", "bundle_consumer.py", "bundle_consumer.sh"):
    check(f"bundle_tool_{f}_is_pin", sha(b / "tools" / f) == sha(bsrc / "scripts/release" / f), f)
if role in ("short", "mini_short", "prime_swiss"):
    check("bundle_unprimed", not (b / "jax/aot").exists() and not (b / "sealed").exists(), f"{b} already primed/sealed")
    seed = Path(man["autotune_seed"]["list"])
    check("seed_list", sha(seed) == man["autotune_seed"]["list_sha256"], str(seed))
    tdir = b / "jax/xla_gpu_per_fusion_autotune_cache_dir"
    have = sorted(f"{sha(p)}  ./{p.relative_to(tdir).as_posix()}" for p in tdir.rglob("*") if p.is_file())
    want = sorted(l for l in seed.read_text().splitlines() if l.strip())
    check("seed_content", have == want, f"{len(have)} files vs {len(want)} seeded")
elif role in ("prime_prod", "prime_wn3", "warm"):
    resume_seal=role=="warm" and (b/"sealed/BUNDLE_MANIFEST.json").is_file()
    check("prime_cache_unsealed", not (b / "sealed").exists() or resume_seal, str(b))
    required = {"prime_prod": ["A1_swiss"], "prime_wn3": ["A1_swiss", "A2_prod", "A3_wn3_0227_3h"], "warm": ["A1_swiss", "A2_prod", "A3_wn3_0227_3h", "A4_wn3_0227_icloud0_3h"]}[role]
    for name in required:
        record = b / "meta/primes" / (name + ".json")
        check("prior_prime_"+name, record.is_file() and json.loads(record.read_text()).get("pin")==pin and json.loads(record.read_text()).get("mode")==mode, str(record))
    if role == 'prime_wn3':
        for name in required:
            try: binding = paid_prime(b/'meta/primes'/(name+'.json'), pin, mode)
            except (AssertionError, OSError, ValueError, KeyError, TypeError) as exc:
                refuse('paid prime '+name+' binding: '+type(exc).__name__)
            check('paid_prime_binding_'+name, True, binding)
        if mode == 'gpu':
            before = b/'meta/E41_CACHE_INTEGRITY.json'
            carried = subprocess.run([sys.executable,'-I',str(tools/'cache_integrity.py'),'check',str(b/'jax'),str(before),str(Path(out_json).with_suffix('.paid_cache.json'))],capture_output=True,text=True)
            check('paid_A1_A2_A3_cache_unchanged', carried.returncode == 0, carried.stdout.strip())
    if role == 'warm' and mode == 'gpu':
        carried = json.loads((tools/'CARRIED_SEAL.json').read_text())
        check('reviewed_paid_seal_present', resume_seal, 'R4 requires the already completed R3 seal; never rebuild')
        check('paid_seal_manifest', sha(b/'sealed/BUNDLE_MANIFEST.json')==carried['manifest_sha256'], str(b/'sealed/BUNDLE_MANIFEST.json'))
        check('paid_seal_tar', sha(b/'sealed/gpuwrf-bundle.tar.gz')==carried['bundle_sha256'], carried['bundle_sha256'])
        tup = json.loads((b/'sealed/BUNDLE_MANIFEST.json').read_text())['tuple']
        check('paid_seal_source', tup==carried['tuple'] and tup['commit']==pin and tup['src_tree']==tree, tup)
    if role=="warm":
        check("E41_before_replay", (b / "meta/E41_POST.json").is_file(), str(b))
        check("E41_same_mode", (b / "meta/E41_MODE").read_text().strip()==mode, str(b))
        check("E41_native_hard_gate", json.loads((b / "meta/E41_RC_POLICY.json").read_text()).get("hard_gate")=="PASS", str(b))
        integrity=b / 'meta/E41_CACHE_INTEGRITY.json'
        check('E41_unsealed_integrity_record',integrity.is_file(),str(integrity))
        integrity_check=subprocess.run([sys.executable,'-I',str(tools/'cache_integrity.py'),'check',str(b/'jax'),str(integrity),str(Path(out_json).with_suffix('.cache_integrity.json'))],capture_output=True,text=True)
        check('E41_unsealed_cache_unchanged',integrity_check.returncode==0,integrity_check.stdout.strip())
elif mode == "gpu":
    check("bundle_sealed", (b / "sealed/BUNDLE_MANIFEST.json").is_file(), f"{b}/sealed (run the short entry first)")
else:  # CPU smoke: a CPU-primed stand-in bundle may be unsealable; session_rw3 then uses W45R_SMOKE_REPLAY_BUNDLE as N
    res["info"]["bundle_sealed"] = (b / "sealed/BUNDLE_MANIFEST.json").is_file()
envs = dict(W45R_PINNED_SOURCE=man["snapshot"], W45R_PIN_MANIFEST=man_path, W45R_PIN=pin, W45R_TREE=tree, W45R_STANDIN=str(man["standin"]).lower(), W45R_SNAP=man["snapshot"],
            W45R_P=str(b), W45R_PSRC=str(bsrc), W45R_P_OLD=str(old), W45R_CASES=str(cdir),
            W45R_MANIFEST_SHA=res["manifest_sha256"], W45R_SRC_DIVERGED=str(diverged).lower(),
            **{f"W45R_IN_{k}": v for k, v in man["inputs"].items()})
Path(out_env).write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in envs.items()))
res["env"] = envs
res["ok"] = True
Path(out_json).write_text(json.dumps(res, indent=1) + "\n")
print("MANIFEST_OK", role, mode, pin[:12], "bundle", b, "standin", man["standin"], "src_diverged", diverged)
