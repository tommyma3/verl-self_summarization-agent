from __future__ import annotations

import sys
import types

from self_summarization_agent.config import ModelConfig


def test_generation_module_imports_with_installed_transformers() -> None:
    import self_summarization_agent.generation as generation

    assert generation.AutoTokenizer is not None


def test_vllm_generator_sets_gpu_memory_utilization(monkeypatch) -> None:
    import self_summarization_agent.generation as generation

    captured: dict[str, object] = {}

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2, 3]

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_path: str, trust_remote_code: bool = False) -> FakeTokenizer:
            captured["tokenizer_model_path"] = model_path
            captured["tokenizer_trust_remote_code"] = trust_remote_code
            return FakeTokenizer()

    class FakeLLM:
        def __init__(
            self,
            *,
            model: str,
            trust_remote_code: bool,
            tensor_parallel_size: int,
            gpu_memory_utilization: float,
            max_model_len: int | None = None,
        ) -> None:
            captured["llm_kwargs"] = {
                "model": model,
                "trust_remote_code": trust_remote_code,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            }

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    vllm_module = types.ModuleType("vllm")
    vllm_module.LLM = FakeLLM
    vllm_module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setattr(generation, "AutoTokenizer", FakeAutoTokenizer)

    generator = generation.build_generator(
        ModelConfig(
            backend="vllm_offline",
            model_path="model",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.6,
            max_model_len=4096,
        ),
        gpu_memory_utilization=0.7,
    )

    assert isinstance(generator, generation.VLLMGenerator)
    assert captured["llm_kwargs"] == {
        "model": "model",
        "trust_remote_code": False,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 4096,
    }
