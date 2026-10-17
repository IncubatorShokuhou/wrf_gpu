"""Active-stream schema admission, separate from the compiled observer."""


def validate_restart(buffers, version):
    if buffers is None:
        if version is not None:
            raise ValueError("restart regional stream active group is missing (E78)")
        return
    from .device import validate_schema
    validate_schema(buffers, version)
