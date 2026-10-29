"""The public version flag must succeed and track the package version."""
import pytest

from gpuwrf import __version__
from gpuwrf.cli import main


def test_version_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0
    output = capsys.readouterr()
    assert output.out.strip() == f'gpuwrf {__version__}'
    assert output.err == ''
