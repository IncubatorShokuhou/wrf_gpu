"""Regenerate docs/release/RELEASE_NOTES_v<X>.md from release_notes/ with links re-based for the docs site.

usage: python3 tools/docsite/sync_release_notes.py 0.4.1 0.4.2   (run from the repository root)
"""
import os, re, sys, posixpath
root = "."
for ver in sys.argv[1:]:
    src = f"release_notes/RELEASE_NOTES_v{ver}.md"
    dst = f"docs/release/RELEASE_NOTES_v{ver}.md"
    text = open(os.path.join(root, src)).read()
    def fix(m):
        t = m.group(2)
        if re.match(r"^[a-z]+:", t) or t.startswith(("#", "/")):
            return m.group(0)
        path, _, frag = t.partition("#")
        full = posixpath.normpath(posixpath.join("release_notes", path))
        if not full.startswith("docs/release/") and full.startswith("release_notes/RELEASE_NOTES_") \
           and os.path.exists(os.path.join(root, "docs/release", posixpath.basename(full))):
            full = "docs/release/" + posixpath.basename(full)
        new = posixpath.relpath(full, "docs/release") + ("/" if path.endswith("/") else "") + (("#" + frag) if frag else "")
        return f"{m.group(1)}({new})"
    out = re.sub(r"(\]\s*)\(([^)\s]+)\)", fix, text)
    hdr = f"<!-- Copy of release_notes/RELEASE_NOTES_v{ver}.md with links re-based for the docs site. Edit the original, then regenerate this copy. -->\n"
    open(os.path.join(root, dst), "w").write(hdr + out)
