"""Regression for writing through cpu/namelist.input -> protected case input."""
from pathlib import Path
import tempfile

from stage_case import link_assets, write_namelist


def test_foreign_namelist_is_never_followed():
    lane = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix')
    with tempfile.TemporaryDirectory(dir=lane, prefix='stage-guard-') as tmp:
        root = Path(tmp)
        protected = root / 'reference'
        protected.mkdir()
        original = protected / 'namelist.input'
        original.write_bytes(b'original sealed namelist\n')
        binary = protected / 'wrfinput_d01'
        binary.write_bytes(b'read-only input')
        seed = root / 'seed'
        seed.mkdir()
        (seed / 'namelist.input').symlink_to(original)
        (seed / 'wrfinput_d01').symlink_to(binary)
        for name in ['namelist.output', 'wrfout_d01_0', 'wrfrst_d01_0', 'rsl.out.0000', 'wrf.exe', 'd0']:
            (seed / name).symlink_to(original)
        case = root / 'case'
        case.mkdir()
        link_assets(seed, case, lane_root=root)
        write_namelist(case, b'private short run\n', lane_root=root)
        assert original.read_bytes() == b'original sealed namelist\n'
        assert not (case / 'namelist.input').is_symlink()
        assert (case / 'namelist.input').read_bytes() == b'private short run\n'
        assert (case / 'wrfinput_d01').read_bytes() == b'read-only input'
        assert set(p.name for p in case.iterdir()) == {'namelist.input', 'wrfinput_d01'}
        hostile = root / 'hostile'
        hostile.mkdir()
        (hostile / 'namelist.input').symlink_to(original)
        try:
            write_namelist(hostile, b'must not write', lane_root=root)
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing namelist symlink was followed')
        assert original.read_bytes() == b'original sealed namelist\n'


if __name__ == '__main__':
    test_foreign_namelist_is_never_followed()
    print('PASS: foreign namelist preserved; six mutable aliases excluded; hostile alias refused')
