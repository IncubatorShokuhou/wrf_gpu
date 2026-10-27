"""Actual E41 post shell with synthetic scanner transport; no GPU/cache reads."""
import json,os,subprocess,tempfile,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;TOOLS=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5')
with tempfile.TemporaryDirectory(prefix='e41_transition_',dir=OUT) as tmp:
    r=Path(tmp);snap=r/'source';snap.mkdir();envfile=r/'manifest.env';envfile.write_text('')
    shim=r/'python_transport.py';shim.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
a=sys.argv[1:];b=a[1:] if a[:1]==['-I'] else a
if b and Path(b[0]).name=='wave_cubin_gate.py':
 p=Path(b[2]);p.mkdir(parents=True,exist_ok=True)
 g=dict(e41_gate='PASS',hard_stack_gate='PASS',conservative_flags=[],n_entries=1,
  coverage=dict(status='PASS',n_discovered=1,n_scanned=1,aot_programs=1),
  entries=[dict(path='SYNTHETIC_CONTROL',elfs=[dict(max_stack_B=32,stack_over256=[],nonleaf_over1k=[],callgraph_unknown=[])],kinput_transpose_status='PASS')])
 (p/'wave_gate.json').write_text(json.dumps(g));raise SystemExit(0)
if b and Path(b[0]).name=='compile_bundle.py':
 print('BUNDLE_OK missing=0 changed=0 new=0 synthetic');raise SystemExit(0)
# Run the real stdlib-only JSON/policy snippets unchanged.
os.execv(sys.executable,[sys.executable,*a])
''');shim.chmod(0o700)
    lib=r/'lib.sh';lib.write_text(f'W44R_PY={shim}\nW44R_TOOLS={TOOLS}\nw44_quiet() {{ return 1; }}\nw44_log() {{ :; }}\n')
    original=(TOOLS/'e41_post.sh').read_text();script=original.replace('source '+str(TOOLS/'lib.sh'),'source '+str(lib),1);assert script!=original
    rows=[]
    for sealed in (False,True):
        p=r/('sealed_bundle' if sealed else 'unsealed_bundle');(p/'jax').mkdir(parents=True);(p/'meta').mkdir();(p/'tools').mkdir()
        if sealed:(p/'sealed').mkdir();(p/'sealed/BUNDLE_MANIFEST.json').write_text('{}')
        run=r/('sealed_run' if sealed else 'unsealed_run');run.mkdir()
        env={**os.environ,'JAX_PLATFORMS':'cpu','PYTHONDONTWRITEBYTECODE':'1','W44R_RUN':str(run),'W44R_ENVFILE':str(envfile),'W44R_MODE':'gpu','W44R_ENTRY_NAME':'prime_wn3','W44R_RECEIPTS':str(r/'receipts'),'W44R_P':str(p),'W44R_SNAP':str(snap)}
        child=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True,timeout=15)
        assert child.returncode==0,(sealed,child.returncode,child.stderr[-1000:])
        post=json.loads((run/'e41/E41_POST.json').read_text())
        rows.append(dict(bundle_sealed=sealed,scanner_rc=post['all_rc'],raw_scan_gate=post['gates']['all']['e41_gate'],post_rc=child.returncode,check=post['check_after_e41']))
report=dict(scope='Unchanged E41 post body; source-lib subprocess boundary localized; synthetic valid scanner PASS; no real GPU/native cache/model touched',controls=rows,
    actual_order='prime_wn3 snapshot+E41+unchanged byte check -> publishmeta -> warm rechecks same inventory -> SEAL',
    files={n:hashlib.sha256((TOOLS/n).read_bytes()).hexdigest() for n in ('e41_post.sh','manifest_check.py','entry_prime_wn3.sh','session_short.sh')})
(OUT/'E41_TRANSITION.json').write_text(json.dumps(report,indent=2)+'\n')
print('Synthetic scanner PASS: GPU pre-seal post rc0 with exact byte integrity; sealed post rc0.')
