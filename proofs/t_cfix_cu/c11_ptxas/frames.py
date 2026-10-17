"""Print the gpuwrf stack chain of HLO stack_frame_ids: python frames.py HLO_TXT ID [ID...]"""
import re, sys
lines = open(sys.argv[1]).read().split('\n', 400000)[:400000]
def table(name):
    i = lines.index(name) + 1; rows = {}
    while lines[i] and lines[i] not in ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'):
        k, v = lines[i].split(' ', 1); rows[int(k)] = v; i += 1
    return rows
files = {k: v.strip('"') for k, v in table('FileNames').items()}
funcs = {k: v.strip('"') for k, v in table('FunctionNames').items()}
locs = {k: (lambda d: (files[int(d['file_name_id'])], funcs[int(d['function_name_id'])], d['line']))(dict(re.findall(r'(\w+)=(\d+)', v))) for k, v in table('FileLocations').items()}
frames = {k: (lambda d: (int(d['file_location_id']), int(d.get('parent_frame_id', 0))))(dict(re.findall(r'(\w+)=(\d+)', v))) for k, v in table('StackFrames').items()}
for a in sys.argv[2:]:
    fid = int(a); print('frame', fid)
    while fid in frames:
        loc, parent = frames[fid]; f = locs[loc]
        if '/gpuwrf/' in f[0]: print('   ', f[0].split('/gpuwrf/')[-1] + ':' + f[2], f[1])
        if parent == fid or parent == 0: break
        fid = parent
