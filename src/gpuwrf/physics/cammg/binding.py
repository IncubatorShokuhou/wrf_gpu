"""Cold-input shared SNOWH/precipitation fields for the opt-in CAM-MG caller."""
import jax.numpy as jnp
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def bind_state(state,path):
    state=state.ensure_conditional_leaves(mp_physics=11)
    shape=state.rain_acc.shape
    updates={}
    with Dataset(path,'r') as ds:
        for leaf,name,default in (
            ('cam_snowh','SNOWH',np.zeros(shape,np.float32)),
            ('cam_rainnc','RAINNC',np.asarray(state.rain_acc+state.snow_acc+state.graupel_acc+state.ice_acc,np.float32)),
            ('cam_snownc','SNOWNC',np.asarray(state.snow_acc+state.ice_acc,np.float32)),
            ('cam_sr','SR',np.zeros(shape,np.float32))):
            value=np.asarray(ds[name][0],np.float32) if name in ds.variables else default
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError(f'CAM-MG cold input {name} shape/nonfinite differs')
            updates[leaf]=jnp.asarray(value,jnp.float32)
        for leaf,name in (('cam_kvh3d','EXCH_H'),('cam_tke_pbl','TKE_PBL')):
            # Registry DataName for shared TKE is lowercase 'tke_pbl'.
            actual=next((key for key in (name,name.lower()) if key in ds.variables),None)
            if actual is not None:
                value=np.asarray(ds[actual][0],np.float32)
                held=np.asarray(getattr(state,leaf)).copy()
                if value.shape not in (held.shape,state.theta.shape) or not np.isfinite(value).all():
                    raise ValueError(f'CAM-MG cold input {name} shape/nonfinite differs')
                held[:value.shape[0]]=value
                updates[leaf]=jnp.asarray(held,jnp.float32)
    return state.replace(**updates)
