"""Milbrandt-Yau two-moment (WRF mp=9), CPU-oracle-qualified experimental port.

All six number scalars and hail are prognostic. No radius coupling: WRF keeps
has_reqc/has_reqi/has_reqs=0 for this scheme. Native GPU/forecast qualification
is pending; the one-program-per-column kernel has no grid-wide vertical loops.
"""
