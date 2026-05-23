from pathlib import Path
import importlib
import os
import sys

import numpy as np
import torch

from self_summarization_agent.bcplus_backend import RealBrowseCompBackend, _build_searcher_args, _ensure_bc_plus_searcher_imports
from self_summarization_agent.config import RetrievalConfig


def test_ensure_bc_plus_searcher_imports_updates_pythonpath(monkeypatch, tmp_path: Path) -> None:
    bc_plus_root = tmp_path / "bc-plus"
    searcher_root = bc_plus_root / "searcher"
    searcher_root.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != str(searcher_root.resolve())])

    _ensure_bc_plus_searcher_imports(bc_plus_root)

    assert sys.path[0] == str(searcher_root.resolve())
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(searcher_root.resolve()), "/existing"]


def test_build_searcher_args_sets_sdpa_attention_by_default() -> None:
    args = _build_searcher_args(RetrievalConfig(model_name="embedding-model"))

    assert args.attn_implementation == "sdpa"


def test_build_searcher_args_allows_attention_override() -> None:
    args = _build_searcher_args(
        RetrievalConfig(model_name="embedding-model", attn_implementation="eager")
    )

    assert args.attn_implementation == "eager"


def test_to_faiss_numpy_casts_bfloat16_to_float32() -> None:
    bc_plus_root = Path(__file__).resolve().parents[1] / "bc-plus"
    _ensure_bc_plus_searcher_imports(bc_plus_root)
    faiss_searcher = importlib.import_module("searchers.faiss_searcher")

    result = faiss_searcher._to_faiss_numpy(torch.ones((1, 2), dtype=torch.bfloat16))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.ones((1, 2), dtype=np.float32))

def test_backend_truncates_by_characters_when_tokenizer_is_unavailable() -> None:
    backend = RealBrowseCompBackend(
        searcher=object(),
        snippet_max_tokens=5,
        document_max_tokens=5,
    )
    backend.snippet_tokenizer = None

    assert backend._truncate_text("abcdefghij", 5) == "abcde"

