"""w42-bundle042: classify every failing RC-suite node as PRE-EXISTING (fails the same way on the v0.4.1 tag) or NEW.

usage: python3 rc_compare.py RC_OUT_DIR TAG_OUT_DIR OUT_JSON
RC_OUT_DIR = rc_suite.py out dir on the candidate (report.json), TAG_OUT_DIR = rc_suite.py out dir of the SAME failing files
run on a v0.4.1-tag snapshot (same runner/env). Per failing node (FAILED/ERROR in the -ra summary or JUnit):
  PRE-EXISTING  the node also FAILs/ERRORs on the tag
  NEW           the node PASSes (or XFAILs/SKIPs) on the tag -> regression candidate, must be explained
  NEW_TEST      the node does not exist on the tag (added after v0.4.1) -> judge by inspection
  TAG_MISSING   the tag run of that file is missing/incomplete -> not classified
A whole-file failure without node ids (collection error / timeout / kill) is classified by the file's tag outcome.
"""
import json
import sys
from pathlib import Path

rc_out, tag_out, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rep = json.loads((rc_out / "report.json").read_text())


def key(path):
    return path.replace("/", "__").removesuffix(".py")


def junit_id(nodeid):
    """pytest nodeid -> (JUnit classname, name): tests/a/b.py::C::t[x] -> ('tests.a.b.C', 't[x]')."""
    parts = nodeid.split("::")
    if not parts[0].endswith(".py"):   # already JUnit-style 'classname::name' (rc_suite fallback when -ra had no ids)
        return parts[0], parts[-1]
    mod = parts[0].removesuffix(".py").replace("/", ".")
    return ".".join([mod, *parts[1:-1]]), parts[-1]


def tag_nodes(file):
    d = tag_out / key(file)
    rp = d / "result.json"
    if not rp.exists():
        return None
    r = json.loads(rp.read_text())
    outcomes = {}
    if (d / "pytest.xml").exists():
        import xml.etree.ElementTree as ET
        for tc in ET.parse(d / "pytest.xml").getroot().iter("testcase"):
            o = "PASS"
            for ch in tc:
                o = {"failure": "FAIL", "error": "ERROR", "skipped": "SKIP"}.get(ch.tag, o)
                if ch.tag == "skipped" and (ch.get("type") or "") == "pytest.xfail":
                    o = "XFAIL"
            outcomes[(tc.get("classname", ""), tc.get("name", ""))] = o
    for n, _ in r.get("failed", []) + r.get("errors", []):
        outcomes[junit_id(n)] = "FAIL"
    return dict(rc=r["rc"], summary=r["summary"], killed=r.get("killed"), outcomes=outcomes)


rows = []
for f in rep["failing_files"]:
    t = tag_nodes(f["file"])
    nodes = [n for n in f["nodes"] if "::" in n] or [f["file"]]
    for n in nodes:
        if t is None:
            cls = "TAG_MISSING"
        elif "::" not in n:
            cls = "PRE-EXISTING" if (t["rc"] not in (0, 5) or t["killed"]) else "NEW"
        else:
            o = t["outcomes"].get(junit_id(n))
            cls = ("PRE-EXISTING" if o in ("FAIL", "ERROR") else "NEW" if o in ("PASS", "XFAIL", "SKIP")
                   else ("NEW_TEST" if t["rc"] in (0, 1, 5) and not t["killed"] else "TAG_MISSING"))
        rows.append(dict(node=n, file=f["file"], cand_rc=f["rc"], cand_summary=f["summary"], cand_killed=f["killed"],
                         msg=(f["msgs"] or {}).get(n, "")[:300] if isinstance(f.get("msgs"), dict) else "",
                         tag_rc=None if t is None else t["rc"], tag_summary=None if t is None else t["summary"],
                         classification=cls))
counts = {}
for r in rows:
    counts[r["classification"]] = counts.get(r["classification"], 0) + 1
res = dict(candidate_head=rep["head"], candidate_src_tree=rep["src_tree"], rows=rows, counts=counts)
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(counts))
for r in rows:
    print(r["classification"], r["node"], "|", (r["cand_summary"] or "")[:80])
