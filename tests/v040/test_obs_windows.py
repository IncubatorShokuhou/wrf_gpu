"""Original WRF rolling-window/duplicate ingestion across twelve real steps."""
from datetime import datetime
from pathlib import Path
import hashlib,json
import jax.numpy as jnp
import numpy as np
from gpuwrf.assimilation.obs_driver import ObservationInputs,initial_observation_carry,_frame
from gpuwrf.io.obs_domain import read_obs_domain
from gpuwrf.io.obs_namelist import DEFAULTS
from gpuwrf.io.obs_prepare import prepare_reports
from gpuwrf.io.obs_projection import ObservationProjection

ROOT=Path(__file__).resolve().parents[2]/'proofs/s1_obsnudge/full/savepoints'


def test_resident_frame_matches_pristine_window_and_duplicates():
    meta=json.loads((ROOT/'WINDOW_MANIFEST.json').read_text())
    path=ROOT/'windows.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['fixture_sha256']
    reports=read_obs_domain(ROOT/'WINDOW_OBS_DOMAIN101')
    opts=dict(DEFAULTS,**meta['options'])
    data=prepare_reports(reports,ObservationProjection(**meta['projection']),
                         datetime.fromisoformat(meta['start']),opts,18.,42,42)
    inputs=ObservationInputs(**{k:jnp.asarray(data[k]) for k in
        ('times','arrivals','deactivated','rio','rjo','rko','variants','variant_steps','lengths')},
        options=tuple(sorted(opts.items())),max_reports=data['max_reports'],fingerprint=meta['obs_sha256'])
    carry=initial_observation_carry(inputs,(3,4,4));archive=np.load(path)
    previous=None
    for row in meta['frames']:
        step=row['step'];p=f'frame{step:03d}_'
        if step%2:continue  # Original in4dob/errob are cadence-held here.
        lo,count,index,variant,varobs,rko,lengths=_frame(carry,inputs,jnp.int32(step),jnp.float32(row['xtime']))
        lo,count,index,variant,varobs,rko,lengths=map(np.asarray,(lo,count,index,variant,varobs,rko,lengths))
        varobs,rko=varobs.copy(),rko.copy()
        truth=archive[p+'varobs_before'];platform=archive[p+'plfo']
        alive=platform!=99
        # Match report levels by the original time/coordinates. Dead replaced
        # profiles retain buffer slots in WRF but contribute no observations.
        flat=[];flat_rko=[]
        for n in range(int(count)):
            if lengths[n]==0:continue
            source=int(index[n]);length=int(lengths[n])
            match=(archive[p+'time_hours']==np.asarray(inputs.times)[source])&alive
            match&=np.abs(archive[p+'rio']-np.asarray(inputs.rio)[source])<1e-5
            match&=np.abs(archive[p+'rjo']-np.asarray(inputs.rjo)[source])<1e-5
            # Surface and upper-air platforms can coexist at one time/site.
            match&=(archive[p+'nlevs_ob']==length)
            ids=np.flatnonzero(match)
            assert len(ids)==length,(step,source,length,ids)
            actual=varobs[n,:length,:5];expected=truth[ids,:5]
            bound=32*np.finfo(np.float32).eps*(np.abs(expected)+np.asarray([40,40,600,.05,100]))
            assert np.all(np.abs(actual-expected)<=bound),(step,source,actual,expected)
            np.testing.assert_array_equal(rko[n,:length],archive[p+'rko_before'][ids])
            # Feed back ORIGINAL errob mutations; this test isolates read/frame
            # lifetime, whose independent numerical operator has its own oracle.
            varobs[n,:length]=archive[p+'varobs_after'][ids]
            rko[n,:length]=archive[p+'rko_after'][ids]
            flat.extend(ids.tolist())
        assert len(flat)==int(np.count_nonzero(alive)),(step,flat,platform)
        carry=carry.replace(frame_start=jnp.asarray(lo),variants=jnp.asarray(variant),
            varobs=jnp.asarray(varobs),rko=jnp.asarray(rko),read_step=jnp.int32(step),count=jnp.asarray(count))
