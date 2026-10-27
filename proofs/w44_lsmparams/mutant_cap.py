"""Test-only deletion controls; count only genuine assertion failures as kills."""
import os


def pytest_configure():
    kind = os.environ.get("W2_CAP_MUTANT")
    if kind == "ignore_effective":
        from gpuwrf.io import wrfinput_consistency as module
        original = module.validate_fields
        def ignored(*args, **kwargs):
            kwargs["effective_smcmax"] = None
            return original(*args, **kwargs)
        module.validate_fields = ignored
    elif kind == "skip_retained":
        import gpuwrf.surface_params.lsmparams as module
        module.validate_retained_porosity = lambda *args, **kwargs: None
    else:
        raise ValueError("choose W2_CAP_MUTANT=ignore_effective|skip_retained")
