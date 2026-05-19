from pathlib import Path
import sys
import types

from self_summarization_agent.config import load_train_config
from self_summarization_agent import train_rllm


def test_load_rllm_train_config() -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))

    assert config.training.backend == "rllm_verl"
    assert config.rllm.backend == "verl"
    assert config.rllm.logger == ["file"]
    assert config.rllm.gpu_memory_utilization == 0.75
    assert config.runtime.tool_budget > 0


def test_build_trainer_config_sets_vllm_memory_utilization() -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))

    trainer_config = train_rllm.build_trainer_config(config)

    assert trainer_config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.75
    assert trainer_config.actor_rollout_ref.rollout.max_model_len == 65536
    assert trainer_config.data.train_batch_size == 8
    assert trainer_config.rllm.trainer.total_batches == 100


def test_build_trainer_wires_rllm_components(monkeypatch) -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))
    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    rllm_module = types.ModuleType("rllm")
    rllm_module.__path__ = []
    experimental_module = types.ModuleType("rllm.experimental")
    experimental_module.__path__ = []
    trainer_module = types.ModuleType("rllm.experimental.unified_trainer")
    trainer_module.AgentTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "rllm", rllm_module)
    monkeypatch.setitem(sys.modules, "rllm.experimental", experimental_module)
    monkeypatch.setitem(sys.modules, "rllm.experimental.unified_trainer", trainer_module)

    backend_module = types.ModuleType("self_summarization_agent.bcplus_backend")
    backend_module.build_backend = lambda bc_plus_root, retrieval_config: "backend"
    generation_module = types.ModuleType("self_summarization_agent.generation")
    generation_module.build_generator = lambda model_config, *, judge_config=None: "generator"

    judge_module = types.ModuleType("self_summarization_agent.judge")

    class FakeJudge:
        def __init__(self, generator) -> None:
            self.generator = generator

    judge_module.RewardJudge = FakeJudge
    rollout_module = types.ModuleType("self_summarization_agent.rllm_agent")
    rollout_module.build_rllm_rollout = lambda *, config, backend: ("flow", backend)
    dataset_module = types.ModuleType("self_summarization_agent.rllm_dataset")
    dataset_module.build_rllm_tasks = lambda *, bc_plus_root, dataset_config, seed: ["task"]
    evaluator_module = types.ModuleType("self_summarization_agent.rllm_evaluator")
    evaluator_module.build_rllm_evaluator = lambda judge: ("evaluator", judge.generator)

    monkeypatch.setitem(sys.modules, "self_summarization_agent.bcplus_backend", backend_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.generation", generation_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.judge", judge_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.rllm_agent", rollout_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.rllm_dataset", dataset_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.rllm_evaluator", evaluator_module)

    trainer = train_rllm.build_trainer(config)

    assert isinstance(trainer, FakeTrainer)
    assert captured["backend"] == "verl"
    assert captured["agent_flow"] == ("flow", "backend")
    assert captured["evaluator"] == ("evaluator", "generator")
    assert captured["train_dataset"] == ["task"]
    assert captured["config"].rllm.algorithm.adv_estimator == "grpo"
    assert captured["config"].rllm.trainer.logger == ["file"]
    assert captured["config"].actor_rollout_ref.rollout.gpu_memory_utilization == 0.75
