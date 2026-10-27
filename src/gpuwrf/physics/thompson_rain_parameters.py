"""REAL Thompson rain-law initialization from a traced av_r coefficient.

Pristine module_mp_thompson.F:786-788 and :801. Rain table-generation uses
the independent Gunn--Kinzer polynomial; its av_r expressions are comments.
The constants below are the original WRF REAL bits recorded by the existing
proofs/thompson/wrf_constants oracle, not another gamma implementation.
"""
from __future__ import annotations


def rain_parameters(av_r):
    """Return t1_qr_qc (=t1_qr_qi), t2_qr_qi, t2_qr_ev as REAL operands."""
    import jax.numpy as jnp
    f=jnp.float32
    av_r=jnp.asarray(av_r,jnp.float32)
    pi=f(3.1415927410125732)
    am_r=f(523.5988159179688)
    sc3=f(0.8581680655479431)
    crg8=f(720.0000610351562)
    crg9=f(6.)
    crg11=f(2.)
    t1=pi*f(.25)*av_r*crg9
    t2=pi*f(.25)*am_r*av_r*crg8
    evaporation=f(.308)*sc3*jnp.sqrt(av_r)*crg11
    return jnp.stack((t1,t2,evaporation))
