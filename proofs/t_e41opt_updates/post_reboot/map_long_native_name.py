"""Resolve a uniquely truncated native STACK name before frozen HLO mapping."""
from pathlib import Path
import hashlib
import json
import sys

tool = Path(__file__).resolve().parents[2] / 't_e41opt/map_flags.py'
original = tool.read_text()
anchor = '    for name, row in flags.items():\n'
assert original.count(anchor) == 1
resolve = '''    resolved = {}
    for native_name, row in flags.items():
        matches = [name for name in calls if name == native_name or name.startswith(native_name)]
        assert len(matches) == 1, (native_name, matches)
        name = matches[0]
        if name != native_name:
            row['native_reported_stack_name'] = native_name
        resolved.setdefault(name, {}).update(row)
    flags = resolved
'''
source = original.replace(anchor, resolve + anchor)
exec(compile(source, str(tool), 'exec'), {'__file__': str(tool), '__name__': '__main__'})
out = Path(sys.argv[2])
(out / 'mapper_receipt.json').write_text(json.dumps({
    'frozen_tool': str(tool), 'frozen_tool_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'effective_tool_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'only_change': 'uniquely resolve truncated native STACK names to full HLO fusion names',
    'scope': 'native resource scan full-name equality is checked again by the packet builder',
}, indent=2) + '\n')
