import re, sys, collections
def norm(path):
    out=[]
    for line in open(path):
        line=line.strip()
        if not line: continue
        line=re.sub(r'(?<![A-Za-z0-9_.])[a-z]{1,4}(?=[\[:\s,)\]=;]|$)','V',line)
        out.append(line)
    return collections.Counter(out)
a=norm(sys.argv[1]); b=norm(sys.argv[2])
plus=b-a; minus=a-b
print('lines main',sum(a.values()),'cand',sum(b.values()), 'only cand',sum(plus.values()),'only main',sum(minus.values()))
for k,v in plus.most_common(60): print(' +',v,k[:260])
for k,v in minus.most_common(60): print(' -',v,k[:260])
