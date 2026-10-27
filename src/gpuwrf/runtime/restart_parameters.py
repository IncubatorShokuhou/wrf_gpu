"""Host restart bindings for optional traced parameter families."""
from collections.abc import Mapping


def static_namelist(namelist):
    """Keep static configuration exact; tokenize separately bound operands only.

    This restart-only view does not change the AOT key or numerical namelist.
    In particular, names, shapes and dtypes remain authenticated.
    """
    get = namelist.get if isinstance(namelist, Mapping) else lambda key, default=None: getattr(namelist, key, default)
    families = {name: get(name) for name in ("diffusion_parameters", "mynn_parameters", "thompson_parameters") if get(name) is not None}
    topo = get("topo_wind_static")
    if not families and topo is None:
        return namelist
    from dataclasses import replace
    import jax
    # canonical_digest's generic array conversion can materialize a
    # ShapeDtypeStruct as an object array (pointer bytes). Plain schema tuples
    # avoid process-dependent identities without changing the shared AOT key.
    token = lambda value: ("gpuwrf-traced-operand-v1", tuple(int(n) for n in value.shape), str(value.dtype))
    updates = {name: jax.tree.map(token, value) for name, value in families.items()}
    if topo is not None:
        updates["topo_wind_static"] = topo._replace(ctopo=token(topo.ctopo))
    return {**namelist, **updates} if isinstance(namelist, Mapping) else replace(namelist, **updates)


def parameter_digests(namelist):
    """OFF is exactly the legacy identity; ON delegates to the family owner."""
    get = namelist.get if isinstance(namelist, Mapping) else lambda key, default=None: getattr(namelist, key, default)
    if not any(get(family) is not None for family in ("diffusion_parameters", "mynn_parameters", "thompson_parameters")):
        return {}
    from gpuwrf.runtime_parameters import restart_parameter_digests
    return restart_parameter_digests(namelist)


def validate_parameters(recorded, namelist):
    if recorded != parameter_digests(namelist):
        raise ValueError("traced runtime parameter values/schema differ from checkpoint (E78)")
