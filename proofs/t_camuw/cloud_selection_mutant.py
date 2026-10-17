"""pytest plugin: force the actual RRTMG assembler to use the wrong MYNN clouds."""


def pytest_configure(config):
    from gpuwrf.coupling import physics_couplers as pc
    original = pc._rrtmg_column_inputs_impl

    def wrong_selection(*args, **kwargs):
        kwargs["use_mynn_sgs_cloud"] = True
        return original(*args, **kwargs)

    pc._rrtmg_column_inputs_impl = wrong_selection
