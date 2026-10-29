from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'v6_layer_controls';T=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6');O.mkdir(exist_ok=True);rows=[]
for name,reaped,compiled,expected in [('ok',True,0,True),('not_reaped',False,0,False),('new_compile',True,1,False)]:
 layer=O/'layers'/name;arm=O/'runs'/name;(layer/'work/work').mkdir(parents=True,exist_ok=True);(layer/'upper').mkdir();(layer/'upper/inert').write_bytes(b'INERT FIXTURE');(layer/'work/work').chmod(0);arm.mkdir(parents=True,exist_ok=True);(arm/'timer.json').write_text(json.dumps(dict(rc=0,teardown=dict(all_reaped=reaped))));log=arm/'scope.log';log.write_text(f'BUNDLE_CONSUMER rc=0 compiled_new={compiled} aliases_new=0 runtime_new=0 other_new=0 layer_compiled=0 receipt=private\n')
 result=subprocess.run([sys.executable,'-B','-I',str(T/'clean_layer.py'),str(O),str(layer),str(arm),str(log)],capture_output=True,text=True);assert (result.returncode==0)==expected,(name,result.stderr);assert layer.exists()!=expected;rows.append(dict(case=name,rc=result.returncode,expected=expected))
 if not expected:
  (layer/'work/work').chmod(0o700);shutil.rmtree(layer)
# Symlinks to a reference stand-in remain untouched; the layer itself may not be a symlink.
ref=O/'reference_standin';ref.mkdir();(ref/'KEEP').write_bytes(b'KEEP');link=O/'layers/symlink';link.symlink_to(ref,target_is_directory=True)
result=subprocess.run([sys.executable,'-B','-I',str(T/'clean_layer.py'),str(O),str(link),str(O/'runs/ok'),str(O/'runs/ok/scope.log')],capture_output=True,text=True);assert result.returncode!=0 and(ref/'KEEP').read_bytes()==b'KEEP';rows.append(dict(case='symlink_reference',refused=True,preserved=True));link.unlink();(ref/'KEEP').unlink();ref.rmdir()
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,checks=rows,scope='inert own fixtures only; actual guard code',bytecode_or_model_source_changed=False),indent=2)+'\n');print('Layer controls PASS4: opaque empty work cleaned; nonreaped/newcompile/refsymlink retained')
