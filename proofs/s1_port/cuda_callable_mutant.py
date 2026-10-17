"""Genuine dispatch mutant: wrap the CUDA callable; identity test must assert.

pytest -p proofs.s1_port.cuda_callable_mutant tests/test_portability.py::test_cuda_pallas_is_original_callable
"""


def pytest_configure(config):
    from gpuwrf.backends import pallas

    original = pallas.__getattr__

    def mutant(name):
        if name == "pallas_call":
            return pallas._jax_call
        return original(name)

    pallas.__getattr__ = mutant
