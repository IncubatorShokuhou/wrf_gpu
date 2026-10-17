"""Keep only original in4dob/errob frame metadata from the 12-step DM case."""
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('freeze',Path(__file__).with_name('freeze_coupled.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def main(case,out):
    out.mkdir(parents=True,exist_ok=True)
    data={};frames=[]
    keys=('step','xtime','rio','rjo','time_hours','lev_in_ob','nlevs_ob','plfo',
          'varobs_before','varobs_after','rko_before','rko_after')
    for path in sorted(case.glob('obs_probe_*.bin')):
        a=m.read_probe(path);prefix=f'frame{int(a["step"]):03d}_'
        for k in keys:
            v=np.asarray(a[k]);assert np.isfinite(v).all()
            data[prefix+k]=v
        frames.append(dict(step=int(a['step']),xtime=float(a['xtime']),count=int(a['nstat']),
                           raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(frames)==12
    dest=out/'windows.npz';np.savez_compressed(dest,**data)
    shutil.copyfile(case/'OBS_DOMAIN101',out/'WINDOW_OBS_DOMAIN101')
    base=json.loads((Path(__file__).parent/'savepoints/COUPLED_MANIFEST.json').read_text())
    options=dict(base['original_provenance']['options'],obs_twindo=.005)
    (out/'WINDOW_MANIFEST.json').write_text(json.dumps(dict(
        frames=frames,fixture_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        obs_sha256=hashlib.sha256((case/'OBS_DOMAIN101').read_bytes()).hexdigest(),
        namelist_sha256=hashlib.sha256((case/'namelist.input').read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256((case/'wrf.exe').read_bytes()).hexdigest(),
        options=options,projection=base['projection'],start=base['original_provenance']['start'],dt_s=18.),indent=2)+'\n')


if __name__=='__main__':main(*map(lambda p:Path(p).resolve(),sys.argv[1:]))
