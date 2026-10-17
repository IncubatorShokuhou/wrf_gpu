"""Selected revised-MM5 outputs retained between WRF surface/LSM/PBL drivers."""

def surface_handoff(raw, skin_before_lsm):
    values = raw if isinstance(raw, dict) else raw._asdict()
    # YSU/Shin-Hong consume FM/FHH; GBM/MRF/GFS consume raw PSIM/PSIH.
    # Keep both contracts. TSKOLD and GZ1OZ0 belong to the paid surface solve.
    result = {key: values[key] for key in
              ('fm','fh','psim','psih','br','wspd','u10','v10','gz1oz0','rmol')}
    result['tsk'] = skin_before_lsm
    return result


def revised_mm5_surface_step(state, grid, *, land=None):
    """Reuse the qualified REAL SF1 producer and its existing Noah-MP ABI."""
    from gpuwrf.coupling.epsilon_pbl import epsilon_surface_step
    skin_before_lsm = state.t_skin
    state, _, noah, raw = epsilon_surface_step(state, grid, land=land, return_full=True)
    if state.sfc_wspd is not None:
        state = state.replace(sfc_wspd=raw.wspd.astype(state.sfc_wspd.dtype))
    return state, surface_handoff(raw, skin_before_lsm), noah
