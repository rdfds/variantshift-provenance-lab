from pathlib import Path

from variantshift.final_freeze import _code_snapshot
from variantshift.provenance import sha256_file


def test_code_snapshot_is_byte_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "workflow").mkdir()
    (root / "configs").mkdir()
    (root / "containers").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    assert _code_snapshot(root, first) == _code_snapshot(root, second)
    assert sha256_file(first) == sha256_file(second)
