"""Actual private Thompson38 bindings for the AOT table-key provider.

This host-only snapshot performs no table casts or device operations. The
provider also registers the three cached NPZ loaders separately, allowing the
common digest to deduplicate their actual arrays within one key construction.
"""
import types


def _constants(values):
    return {name: value for name, value in values.items()
            if name.isupper() and not callable(value)
            and not isinstance(value, (type, types.ModuleType))}


def resolved_bindings():
    from . import native

    result = {"native_aerosol_table": native.THOMPSON_AERO_TABLES}
    # FunctionType copies own separate __globals__ dictionaries. Namespace
    # attributes and function globals may be rebound independently; both can
    # reach the product. Never use object IDs in the persistent snapshot.
    original_modules = {"gpuwrf.physics.thompson_column",
                        "gpuwrf.physics.thompson_aero_column"}
    seen = set()
    for label, namespace in (("TC", native.TC), ("AERO", native.AERO)):
        values = vars(namespace)
        result[label + "_attributes"] = _constants(values)
        for name, value in sorted(values.items()):
            function = getattr(value, "__wrapped__", value)
            if not isinstance(function, types.FunctionType):
                continue
            if function.__module__ not in original_modules:
                continue
            result[(label, name, "function")] = (
                function.__module__, function.__qualname__,
                function.__defaults__, function.__kwdefaults__,
                tuple(cell.cell_contents for cell in function.__closure__ or ()),
            )
            env = function.__globals__
            if id(env) not in seen:
                seen.add(id(env))  # Local traversal deduplication only.
                result[(label, name, "globals")] = _constants(env)
    return result
