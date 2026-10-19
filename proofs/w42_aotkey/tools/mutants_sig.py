import sys
from pathlib import Path
p = Path("src/gpuwrf/runtime/aot_key_cli.py")
s = Path(sys.argv[2]).read_text()
m = sys.argv[1]
table = {
 "S1_not_installed": ('    with tempfile.TemporaryDirectory(prefix="gpuwrf_aot_key_") as tmp, removed_on_signal(Path(tmp)):',
                      '    with tempfile.TemporaryDirectory(prefix="gpuwrf_aot_key_") as tmp:'),
 "S2_no_delete": ('            shutil.rmtree(path, ignore_errors=True)\n', '            pass\n'),
 "S3_exit_1": ('        signal.signal(signum, signal.SIG_DFL)\n        os.kill(os.getpid(), signum)\n', '        os._exit(1)\n'),
 "S4_no_restore": ('        for signum, old in previous.items():\n            signal.signal(signum, old)\n', '        pass\n'),
}
a, b = table[m]
assert s.count(a) == 1, m
p.write_text(s.replace(a, b))
