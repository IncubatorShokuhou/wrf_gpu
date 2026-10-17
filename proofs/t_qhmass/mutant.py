"""A genuine deletion mutant: remove only hail from the shared mass selector."""
import sys
import pytest


class DeleteHail:
    def pytest_sessionstart(self, session):
        from gpuwrf.contracts import moisture
        self.original = moisture._OPTIONAL_MOIST
        moisture._OPTIONAL_MOIST = tuple(n for n in self.original if n != "qh")

    def pytest_sessionfinish(self, session, exitstatus):
        from gpuwrf.contracts import moisture
        moisture._OPTIONAL_MOIST = self.original


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[DeleteHail()]))
