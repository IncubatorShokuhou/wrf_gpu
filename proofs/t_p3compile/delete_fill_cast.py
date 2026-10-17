"""Delete the real native fill cast; require the typed-IR gate to reject it."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from gpuwrf.kernels import _p3_memory as memory
    source = inspect.getsource(memory._load_triton)
    needle = 'tl._ir_cast(_operand(other, other_aval, shape), f32, signed=False)'
    assert source.count(needle) == 1
    mutant = source.replace(needle, '_operand(other, other_aval, shape)')
    # Re-register only this process's actual production lowering rule.
    exec(compile(mutant, '<delete_p3_fill_cast>', 'exec'), memory.__dict__)
    root = Path(__file__).resolve().parents[2]
    sys.argv = ['cubin_probe.py', '--target', str(a.target),
                '--out', str(a.out / 'native')]
    start = time.monotonic()
    try:
        runpy.run_path(str(root / 'proofs/t_p3compile/cubin_probe.py'), run_name='__main__')
    except AssertionError as exc:
        assert str(exc) == 'P3 inline load operands must use typed REAL4 fill', str(exc)
        report = {'scope': 'real native fill-cast deletion, no GPU',
                  'assertions': 1, 'setup_errors': 0, 'killed': True,
                  'reason': str(exc), 'wall_s': time.monotonic() - start,
                  'mutant_source_sha256': hashlib.sha256(mutant.encode()).hexdigest()}
        (a.out / 'deletion.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return
    raise AssertionError('Native fill-cast deletion survived the actual typed-IR gate')


if __name__ == '__main__':
    main()
