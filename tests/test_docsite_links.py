"""User-guide links (tools/docsite/build.py): GitHub Pages serves docs/ as plain files, so a .md target
must point at the rendered GitHub page and nothing may leave the Pages root; a missing file target or
in-guide anchor fails the build."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_module():
    spec = importlib.util.spec_from_file_location("docsite_build", ROOT / "tools" / "docsite" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guide_builds_with_rendered_markdown_links(tmp_path):
    _build_module().build(tmp_path)
    pages = sorted(tmp_path.glob("*.html"))
    assert len(pages) > 15
    for page in pages:
        rel = [h for h in re.findall(r'href="([^"]*)"', page.read_text())
               if h and not re.match(r"[a-z][a-z0-9+.-]*:|#|/", h)]
        assert not [h for h in rel if h.split("#")[0].endswith(".md")], page.name
        assert not [h for h in rel if h.startswith("../")], page.name


def test_rewrite_links_reports_missing_targets_and_anchors():
    build = _build_module()
    problems = []
    out = build.rewrite_links(
        '<a href="b.html#y">1</a><a href="b.html#nope">2</a><a href="#x">3</a><a href="#z">4</a>'
        '<a href="onboarding.md#install-the-container">5</a><a href="../KNOWN_ISSUES.md">6</a>'
        '<a href="no_such_guide.md">7</a>',
        "a", ROOT, {"a": {"x"}, "b": {"y"}}, problems)
    assert 'href="https://github.com/wrf-gpu/wrf_gpu/blob/main/docs/onboarding.md#install-the-container"' in out
    assert 'href="https://github.com/wrf-gpu/wrf_gpu/blob/main/KNOWN_ISSUES.md"' in out
    assert 'href="b.html#y"' in out
    assert sorted(problems) == ["a: missing anchor #z", "a: missing anchor b.html#nope",
                                "a: missing target no_such_guide.md"]
