from __future__ import annotations


def test_generation_module_imports_with_installed_transformers() -> None:
    import self_summarization_agent.generation as generation

    assert generation.AutoTokenizer is not None

