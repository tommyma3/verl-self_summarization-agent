from pathlib import Path
import os
import sys
import types

from self_summarization_agent.config import load_train_config
from self_summarization_agent import train_rllm


def test_load_rllm_train_config() -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))

    assert config.training.backend == "rllm_verl"
    assert config.rllm.backend == "verl"
    assert config.rllm.logger == ["file"]
    assert config.rllm.gpu_memory_utilization == 0.80
    assert config.runtime.tool_budget > 0


def test_build_trainer_config_sets_vllm_memory_utilization(monkeypatch) -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))
    monkeypatch.setenv(train_rllm.CUDA_VISIBLE_DEVICES_ENV, "0,1")

    trainer_config = train_rllm.build_trainer_config(config)

    assert trainer_config.actor_rollout_ref.rollout.gpu_memory_utilization == config.rllm.gpu_memory_utilization
    assert trainer_config.actor_rollout_ref.rollout.name == "vllm"
    assert trainer_config.actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config == "vllm"
    assert trainer_config.actor_rollout_ref.actor.fsdp_config.model_dtype == "bfloat16"
    assert trainer_config.actor_rollout_ref.ref.fsdp_config.model_dtype == "bfloat16"
    assert trainer_config.actor_rollout_ref.rollout.max_model_len == config.rollout.max_model_len
    assert trainer_config.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert trainer_config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert trainer_config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert trainer_config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert trainer_config.data.train_batch_size == 8
    assert trainer_config.trainer.n_gpus_per_node == 2
    assert trainer_config.trainer.cuda_visible_devices == "0,1"
    assert trainer_config.rllm.trainer.total_batches == 100
    assert trainer_config.rllm.workflow.retry_limit == 3


def test_configure_vllm_runtime_environment_disables_dynamic_lora(monkeypatch) -> None:
    monkeypatch.delenv(train_rllm.VLLM_RUNTIME_LORA_ENV, raising=False)

    train_rllm.configure_vllm_runtime_environment()

    assert os.environ[train_rllm.VLLM_RUNTIME_LORA_ENV] == "false"


def test_configure_vllm_runtime_environment_preserves_user_override(monkeypatch) -> None:
    monkeypatch.setenv(train_rllm.VLLM_RUNTIME_LORA_ENV, "true")

    train_rllm.configure_vllm_runtime_environment()

    assert os.environ[train_rllm.VLLM_RUNTIME_LORA_ENV] == "true"


def test_configure_cuda_visible_devices_preserves_existing_mask(monkeypatch) -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))
    monkeypatch.setenv(train_rllm.CUDA_VISIBLE_DEVICES_ENV, "3,4")

    train_rllm.configure_cuda_visible_devices(config)

    assert os.environ[train_rllm.CUDA_VISIBLE_DEVICES_ENV] == "3,4"


def test_configure_cuda_visible_devices_uses_training_gpu_ids(monkeypatch) -> None:
    config = load_train_config(
        Path("configs/train/rllm_verl.yaml"), {"training": {"gpu_ids": [2, 5]}}
    )
    monkeypatch.delenv(train_rllm.CUDA_VISIBLE_DEVICES_ENV, raising=False)

    train_rllm.configure_cuda_visible_devices(config)

    assert os.environ[train_rllm.CUDA_VISIBLE_DEVICES_ENV] == "2,5"


def test_configure_cuda_visible_devices_uses_driver_device_count(monkeypatch) -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))
    monkeypatch.delenv(train_rllm.CUDA_VISIBLE_DEVICES_ENV, raising=False)
    monkeypatch.setattr(train_rllm, "_driver_cuda_device_count", lambda: 3)

    train_rllm.configure_cuda_visible_devices(config)

    assert os.environ[train_rllm.CUDA_VISIBLE_DEVICES_ENV] == "0,1,2"


def test_configure_dependency_warning_filters_sets_worker_options(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONWARNINGS", "default::UserWarning")

    train_rllm.configure_dependency_warning_filters()

    options = os.environ["PYTHONWARNINGS"].split(",")
    assert options[0] == "default::UserWarning"
    assert "ignore:The cuda.nvrtc module is deprecated:FutureWarning" in options
    assert "ignore:FSDP.state_dict_type:FutureWarning:verl.workers.engine.fsdp.transformer_impl" in options


def test_configure_dependency_warning_filters_does_not_duplicate_options(monkeypatch) -> None:
    existing_option = "ignore:The cuda.nvrtc module is deprecated:FutureWarning"
    monkeypatch.setenv("PYTHONWARNINGS", existing_option)

    train_rllm.configure_dependency_warning_filters()
    train_rllm.configure_dependency_warning_filters()

    options = os.environ["PYTHONWARNINGS"].split(",")
    assert options.count(existing_option) == 1


def test_ensure_worker_library_paths_adds_venv_shared_libraries(monkeypatch, tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    cuda_lib = site_packages / "nvidia" / "cu13" / "lib"
    torch_lib = site_packages / "torch" / "lib"
    cuda_lib.mkdir(parents=True)
    torch_lib.mkdir(parents=True)
    monkeypatch.setattr(train_rllm.site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(train_rllm.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    train_rllm.ensure_worker_library_paths()

    paths = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(cuda_lib) in paths
    assert str(torch_lib) in paths
    assert paths[-1] == "/existing"


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
    generation_module = types.ModuleType("self_summarization_agent.generation")
    def fake_build_generator(model_config, *, judge_config=None, gpu_memory_utilization=None):
        captured["judge_gpu_memory_utilization"] = gpu_memory_utilization
        return "generator"

    generation_module.build_generator = fake_build_generator

    judge_module = types.ModuleType("self_summarization_agent.judge")

    class FakeJudge:
        def __init__(self, generator) -> None:
            self.generator = generator

    judge_module.RewardJudge = FakeJudge
    rollout_module = types.ModuleType("self_summarization_agent.rllm_agent")
    rollout_module.build_rllm_rollout = lambda *, config, backend_factory: ("flow", backend_factory())
    backend_module.build_backend = lambda bc_plus_root, retrieval_config: ("backend", bc_plus_root, retrieval_config.backend)
    dataset_module = types.ModuleType("self_summarization_agent.rllm_dataset")
    dataset_module.build_rllm_dataset = (
        lambda *, bc_plus_root, dataset_config, seed, tokenizer_path, max_query_tokens, trust_remote_code, name, split: (
            "dataset",
            name,
            split,
            tokenizer_path,
            max_query_tokens,
            trust_remote_code,
        )
    )
    evaluator_module = types.ModuleType("self_summarization_agent.rllm_evaluator")
    evaluator_module.build_rllm_evaluator = lambda *, judge_factory: ("evaluator", judge_factory().generator)

    monkeypatch.setitem(sys.modules, "self_summarization_agent.bcplus_backend", backend_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.generation", generation_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.judge", judge_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.rllm_agent", rollout_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.rllm_dataset", dataset_module)
    monkeypatch.setitem(sys.modules, "self_summarization_agent.rllm_evaluator", evaluator_module)

    called: list[str] = []
    monkeypatch.setattr(train_rllm, "configure_cuda_visible_devices", lambda config: called.append("cuda"))
    monkeypatch.setattr(train_rllm, "configure_dependency_warning_filters", lambda: called.append("warnings"))

    trainer = train_rllm.build_trainer(config)

    assert called == ["cuda", "warnings"]
    assert isinstance(trainer, FakeTrainer)
    assert captured["backend"] == "verl"
    assert captured["agent_flow"] == ("flow", ("backend", "bc-plus", "bm25"))
    assert captured["evaluator"] == ("evaluator", "generator")
    assert captured["train_dataset"] == (
        "dataset",
        "qwen-bcplus-rllm-rllm",
        "train",
        config.model.model_path,
        config.rllm.max_prompt_length - 1024,
        config.model.trust_remote_code,
    )
    assert captured["val_dataset"] == captured["train_dataset"]
    assert captured["config"].rllm.algorithm.adv_estimator == "grpo"
    assert captured["config"].rllm.trainer.logger == ["file"]
    assert captured["config"].actor_rollout_ref.rollout.gpu_memory_utilization == config.rllm.gpu_memory_utilization
    assert captured["judge_gpu_memory_utilization"] == config.rllm.gpu_memory_utilization
