"""Original :2168-2176 predicate, exposed by an admissible tiny Nt value."""
from collections import namedtuple
import inspect
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics.thompson_nt_c import cloud_parameters


def test_inactive_cloud_keeps_original_D0c_and_guard_is_load_bearing(monkeypatch):
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL","1")
    monkeypatch.setenv("GPUWRF_THOMPSON_WRF_CONSTANTS","1")
    # Exercise the actual namespace/guard. Only Ref reads are represented by
    # plain arrays here; no freezing/other table consumer is called.
    monkeypatch.setattr(full,"_load",lambda ref,index,**kw:ref[index])
    Cold=namedtuple("Cold",("tpi_qcfz","tni_qcfz"))
    dummy=full._Table(jnp.zeros((1,1,1),jnp.float64),jnp.array(True))
    nt=jnp.float32(2)
    env=full._parameter_namespace(tc,tc.THOMPSON_TABLES,Cold(dummy,dummy),4,4,
                                  ("nt_c","__nt_c_cloud"),(nt,cloud_parameters(nt)))
    qc=jnp.asarray([0.,.5*tc.R1,tc.R1,2*tc.R1],jnp.float32)
    rho=jnp.ones_like(qc)
    result=env["_cloud_distribution"](qc,rho)
    expected=np.float32(tc.D0C)
    assert np.isfinite(np.asarray(result[3])).all()
    np.testing.assert_array_equal(np.asarray(result[3])[:3],np.full(3,expected))
    assert np.asarray(result[3])[3]>expected
    # Removing just the original L_qc predicate restores a ghost collector:
    # mvd>D0c despite qc==0. The active cell remains unchanged.
    removed=inspect.getclosurevars(env["_cloud_distribution"]).nonlocals["original_distribution"](qc,rho)
    assert (np.asarray(removed[3])[:3]>expected).all()
    assert np.asarray(removed[3])[3]==np.asarray(result[3])[3]
