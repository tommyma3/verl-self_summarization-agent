from pathlib import Path
import os
import sys

from self_summarization_agent.bcplus_backend import _ensure_bc_plus_searcher_imports


def test_ensure_bc_plus_searcher_imports_updates_pythonpath(monkeypatch, tmp_path: Path) -> None:
    bc_plus_root = tmp_path / "bc-plus"
    searcher_root = bc_plus_root / "searcher"
    searcher_root.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != str(searcher_root.resolve())])

    _ensure_bc_plus_searcher_imports(bc_plus_root)

    assert sys.path[0] == str(searcher_root.resolve())
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(searcher_root.resolve()), "/existing"]
