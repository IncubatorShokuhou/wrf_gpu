"""Sequential production CPU arms in one JAX process (E105/E119)."""
from pathlib import Path
import os,sys,json
from smoke import main

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
arms={2:'<USER_HOME>/wrf_gpu2_lanes/s2-sfdda/SF18-original/mode2',
      1:'<USER_HOME>/wrf_gpu2_lanes/s2-sfdda/SF17-original/mode1',
      0:'<USER_HOME>/wrf_gpu2_lanes/s2-sfdda/SF17-original/mode0'}
for mode in tuple(map(int,sys.argv[2:])) or (2,1,0):
    os.environ['SF_ORIGINAL_CASE']=arms[mode]
    main(root/f'mode{mode}',mode)
    result=json.loads((root/f'mode{mode}/result.json').read_text())
    print(json.dumps({'mode':mode,'wall_s':result['wall_s'],'restart_leaves':result['restart_exact_leaves'],
                      'hard_field_failures':result['hard_field_failures'],
                      'guards':[r['guards'] for r in result['steps']]}),flush=True)
