from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import sys
from typing import Any
import warnings

from self_summarization_agent.config import load_train_config, parse_cli_overrides


RLLM_CONFIG_MODULE = "rllm.experimental.config"
RLLM_CONFIG_NAME = "unified"
VLLM_RUNTIME_LORA_ENV = "VLLM_ALLOW_RUNTIME_LORA_UPDATING"
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
SUPPRESSED_FUTURE_WARNING_FILTERS = (
    (
        "The cuda.nvrtc module is deprecated",
        "",
    ),
    (
        "FSDP.state_dict_type",
        "verl.workers.engine.fsdp.transformer_impl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-summarization agent with rLLM/verl.")
    parser.add_argument("--config", required=True, help="Path to an rLLM training config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values as key=value.")
    return parser.parse_args()


def _count_cuda_visible_devices(value: str | None) -> int | None:
    if value is None:
        return None
    devices = [device.strip() for device in value.split(",") if device.strip()]
    return len(devices)


def _driver_cuda_device_count() -> int:
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def configured_gpu_count(config: Any) -> int:
    gpu_ids = list(getattr(config.training, "gpu_ids", []) or [])
    if gpu_ids:
        return len(gpu_ids)

    visible_count = _count_cuda_visible_devices(os.environ.get(CUDA_VISIBLE_DEVICES_ENV))
    if visible_count is not None:
        return visible_count

    return _driver_cuda_device_count()


def configure_cuda_visible_devices(config: Any) -> None:
    """Ensure Ray/vLLM worker processes inherit usable CUDA visibility."""
    existing_visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES_ENV)
    if existing_visible_devices:
        return

    gpu_ids = list(getattr(config.training, "gpu_ids", []) or [])
    if gpu_ids:
        os.environ[CUDA_VISIBLE_DEVICES_ENV] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        return

    device_count = _driver_cuda_device_count()
    if device_count <= 0:
        raise RuntimeError(
            "rLLM/verl training requires CUDA devices, but the driver process cannot see any. "
            "Set CUDA_VISIBLE_DEVICES or run on a GPU node before starting training."
        )
    os.environ[CUDA_VISIBLE_DEVICES_ENV] = ",".join(str(index) for index in range(device_count))


def build_trainer_config(config: Any) -> Any:
    try:
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
    except ImportError as exc:
        raise ImportError(
            "Hydra and OmegaConf are required to compose the rLLM/verl trainer config."
        ) from exc

    if GlobalHydra.instance().is_initialized():
        trainer_config = compose(config_name=RLLM_CONFIG_NAME)
    else:
        with initialize_config_module(config_module=RLLM_CONFIG_MODULE, version_base=None):
            trainer_config = compose(config_name=RLLM_CONFIG_NAME)

    rollout_n = config.rllm.rollout_group_size
    sampling = {
        "temperature": config.rollout.temperature,
        "top_p": config.rollout.top_p,
        "top_k": -1,
        "max_tokens": config.rllm.max_response_length,
    }
    updates = {
        "model.name": config.model.model_path,
        "data.train_batch_size": config.rllm.train_batch_size,
        "data.max_prompt_length": config.rllm.max_prompt_length,
        "data.max_response_length": config.rllm.max_response_length,
        "data.trust_remote_code": config.model.trust_remote_code,
        "actor_rollout_ref.model.path": config.model.model_path,
        "actor_rollout_ref.model.trust_remote_code": config.model.trust_remote_code,
        "actor_rollout_ref.actor.ppo_mini_batch_size": config.rllm.ppo_mini_batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": config.rllm.ppo_micro_batch_size_per_gpu,
        "actor_rollout_ref.actor.optim.lr": config.training.learning_rate,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": config.model.dtype,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": config.model.dtype,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": config.rllm.ppo_micro_batch_size_per_gpu,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.dtype": config.model.dtype,
        "actor_rollout_ref.rollout.gpu_memory_utilization": config.rllm.gpu_memory_utilization,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": config.rllm.ppo_micro_batch_size_per_gpu,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": config.rollout.tensor_parallel_size,
        "actor_rollout_ref.rollout.max_model_len": config.rollout.max_model_len,
        "actor_rollout_ref.rollout.prompt_length": config.rllm.max_prompt_length,
        "actor_rollout_ref.rollout.response_length": config.rllm.max_response_length,
        "actor_rollout_ref.rollout.temperature": config.rollout.temperature,
        "actor_rollout_ref.rollout.top_p": config.rollout.top_p,
        "actor_rollout_ref.rollout.do_sample": config.rollout.do_sample,
        "actor_rollout_ref.rollout.n": rollout_n,
        "actor_rollout_ref.rollout.val_kwargs.temperature": config.judge.temperature,
        "actor_rollout_ref.rollout.val_kwargs.top_p": config.judge.top_p,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": config.judge.do_sample,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config": "vllm",
        "rllm.backend": config.rllm.backend,
        "rllm.algorithm.adv_estimator": config.rllm.algorithm,
        "rllm.rollout.n": rollout_n,
        "rllm.rollout.n_val": 1,
        "rllm.rollout.train": sampling,
        "rllm.rollout.val": {
            "temperature": config.judge.temperature,
            "top_p": config.judge.top_p,
            "top_k": -1,
            "max_tokens": config.judge.max_new_tokens,
        },
        "rllm.workflow.n_parallel_tasks": config.rollout.max_concurrent_episodes,
        "rllm.workflow.retry_limit": 3,
        "trainer.n_gpus_per_node": configured_gpu_count(config),
        "trainer.cuda_visible_devices": os.environ.get(CUDA_VISIBLE_DEVICES_ENV),
        "rllm.trainer.logger": config.rllm.logger,
        "rllm.trainer.project_name": config.rllm.project_name,
        "rllm.trainer.experiment_name": config.rllm.experiment_name,
        "rllm.trainer.save_freq": config.rllm.save_freq,
        "rllm.trainer.test_freq": config.rllm.eval_freq,
        "rllm.trainer.total_batches": config.rllm.total_training_steps,
    }
    with open_dict(trainer_config):
        for key, value in updates.items():
            if value is not None:
                OmegaConf.update(trainer_config, key, value, merge=False)
    return trainer_config


def configure_vllm_runtime_environment() -> None:
    """Keep rLLM Ray defaults from enabling development-only vLLM LoRA APIs."""
    os.environ.setdefault(VLLM_RUNTIME_LORA_ENV, "false")


def _python_warning_option(message: str, module: str) -> str:
    parts = ["ignore", message, "FutureWarning"]
    if module:
        parts.append(module)
    return ":".join(parts)


def configure_dependency_warning_filters() -> None:
    """Suppress known third-party deprecation warnings in Ray worker logs."""
    existing_options = [
        option
        for option in os.environ.get("PYTHONWARNINGS", "").split(",")
        if option
    ]
    updated_options = list(existing_options)
    for message, module in SUPPRESSED_FUTURE_WARNING_FILTERS:
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=FutureWarning,
            module=module,
        )
        option = _python_warning_option(message, module)
        if option not in updated_options:
            updated_options.append(option)

    if updated_options:
        os.environ["PYTHONWARNINGS"] = ",".join(updated_options)


def ensure_worker_library_paths() -> None:
    """Expose venv CUDA/Torch shared libraries to Ray worker processes."""
    candidate_dirs: list[str] = []
    for root in [*site.getsitepackages(), str(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")]:
        site_root = Path(root)
        candidate_dirs.extend(str(path) for path in site_root.glob("nvidia/**/lib") if path.is_dir())
        torch_lib = site_root / "torch" / "lib"
        if torch_lib.is_dir():
            candidate_dirs.append(str(torch_lib))

    existing = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    updated = [path for path in candidate_dirs if path not in existing]
    if updated:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*updated, *existing])


def build_reward_judge(
    *,
    model_config: Any,
    judge_config: Any,
    gpu_memory_utilization: float | None,
) -> Any:
    from self_summarization_agent.generation import build_generator
    from self_summarization_agent.judge import RewardJudge

    return RewardJudge(
        build_generator(
            model_config,
            judge_config=judge_config,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    )


def build_search_backend(
    *,
    bc_plus_root: str,
    retrieval_config: Any,
) -> Any:
    from self_summarization_agent.bcplus_backend import build_backend

    return build_backend(bc_plus_root, retrieval_config)


def build_trainer(config: Any) -> Any:
    configure_cuda_visible_devices(config)
    configure_vllm_runtime_environment()
    configure_dependency_warning_filters()
    ensure_worker_library_paths()

    try:
        from rllm.experimental.unified_trainer import AgentTrainer
    except ImportError as exc:
        raise ImportError(
            "rLLM is required for train_rllm.py. Install the rllm extra in the training environment."
        ) from exc

    from self_summarization_agent.rllm_agent import build_rllm_rollout
    from self_summarization_agent.rllm_dataset import build_rllm_dataset
    from self_summarization_agent.rllm_evaluator import build_rllm_evaluator

    train_dataset = build_rllm_dataset(
        bc_plus_root=config.experiment.bc_plus_root,
        dataset_config=config.dataset,
        seed=config.experiment.seed,
        tokenizer_path=config.model.model_path,
        max_query_tokens=max(1, config.rllm.max_prompt_length - 1024),
        trust_remote_code=config.model.trust_remote_code,
        name=f"{config.experiment.name}-rllm",
        split="train",
    )
    agent_flow = build_rllm_rollout(
        config=config,
        backend_factory=lambda: build_search_backend(
            bc_plus_root=config.experiment.bc_plus_root,
            retrieval_config=config.retrieval,
        ),
    )
    evaluator = build_rllm_evaluator(
        judge_factory=lambda: build_reward_judge(
            model_config=config.model,
            judge_config=config.judge,
            gpu_memory_utilization=config.rllm.gpu_memory_utilization,
        )
    )
    trainer_config = build_trainer_config(config)

    return AgentTrainer(
        backend=config.rllm.backend,
        agent_flow=agent_flow,
        evaluator=evaluator,
        config=trainer_config,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
    )


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    if config.training.backend != "rllm_verl":
        raise ValueError(f"train_rllm.py requires training.backend='rllm_verl', got {config.training.backend!r}")
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
