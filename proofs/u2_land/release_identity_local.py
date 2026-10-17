"""Official s1-port identity algorithm, with its temporary directory in this lane."""
import hashlib,json,sys
from pathlib import Path
tree,out=map(Path,sys.argv[1:3]);official=tree/'proofs/s1_port/release_identity.py'
source=official.read_text()
old='<USER_HOME>/wrf_gpu2_lanes/s1-port';new='<USER_HOME>/wrf_gpu2_lanes/u2-land'
assert source.count(old)==1
adapted=source.replace(old,new)
exec(compile(adapted,str(official),'exec'),{'__file__':str(official),'__name__':'__main__'})
result=json.loads(out.read_text())
result['lane_scratch_adaptation']=dict(official_sha256=hashlib.sha256(source.encode()).hexdigest(),change='temporary directory only',old=old,new=new)
out.write_text(json.dumps(result,indent=2)+'\n')
