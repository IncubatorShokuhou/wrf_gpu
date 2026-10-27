"""Explicit Thompson parameter changes use the existing branch protocol."""
import json

import pytest

from gpuwrf.runtime.checkpoint_branch import branch_identity, load_manifest


def test_thompson_branch_declares_the_exact_changed_value_digest(tmp_path):
    changes = {"schema_version": 1, "parameters": {"thompson_parameters": {"d01": "b" * 64}}}
    path = tmp_path / "branch.json"
    path.write_text(json.dumps(changes))
    assert load_manifest(path) == changes
    parent_identity = {"output_dir": "old", "thompson_parameters": {"d01": "a" * 64}, "static": "preserved"}
    parent = {"branch_schema_version": 1, "identity": parent_identity,
              "domains": ["d01"], "sha256": "c" * 64, "own_steps": {"d01": 10}}
    (tmp_path / "manifest.json").write_text(json.dumps(parent))
    current = {**parent_identity, "output_dir": "new", "thompson_parameters": {"d01": "b" * 64}}
    result = branch_identity(parent, current, changes, tmp_path)
    assert result["thompson_parameters"] == {"d01": "b" * 64}
    assert result["branch"]["parent_identity"] == parent_identity
    with pytest.raises(ValueError, match="undeclared"):
        branch_identity(parent, current, {"schema_version": 1}, tmp_path)
    with pytest.raises(ValueError, match="must already be enabled"):
        branch_identity({**parent, "identity": {"output_dir": "old", "static": "preserved"}},
                        current, changes, tmp_path)
