from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage
from verifiers.v1.utils.loaders import taskset_config_type


class MPSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    taskset: str
    taskset_args: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(1, ge=1)
    num_tasks: int = Field(32, ge=1)
    group_size: int = Field(4, ge=2)
    max_completion_tokens: int = Field(64, ge=1)
    temperature: float = Field(1.0, gt=0)
    top_p: float = Field(1.0, gt=0, le=1)
    learning_rate: float = Field(1e-5, gt=0)
    max_grad_norm: float = Field(1.0, gt=0)
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    lora_rank: int = Field(8, ge=0)
    lora_alpha: int = Field(16, ge=1)
    seed: int = 0
    trust_remote_code: bool = False
    output_dir: Path | None = Path("outputs/mps")


@dataclass
class Rollout:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion: str
    reward: float


def group_advantages(rewards: list[float]) -> list[float]:
    values = torch.tensor(rewards, dtype=torch.float32)
    std = values.std(unbiased=False)
    if std == 0:
        return [0.0] * len(rewards)
    return ((values - values.mean()) / std).tolist()


def completion_logprobs(logits: torch.Tensor, sequence: torch.Tensor, prompt_length: int) -> torch.Tensor:
    completion_length = sequence.shape[1] - prompt_length
    if completion_length < 1:
        raise ValueError("a rollout must contain at least one completion token")
    start = prompt_length - 1
    selected = torch.log_softmax(logits.float(), dim=-1)[:, start : start + completion_length]
    targets = sequence[:, prompt_length:].unsqueeze(-1)
    return selected.gather(-1, targets).squeeze(-1).squeeze(0)


def make_trace(task: vf.Task, completion: str) -> vf.Trace:
    nodes: list[MessageNode] = []

    def append(message, sampled: bool = False) -> None:
        nodes.append(MessageNode(parent=len(nodes) - 1 if nodes else None, message=message, sampled=sampled))

    if task.data.system_prompt:
        append(SystemMessage(content=task.data.system_prompt))
    if isinstance(task.data.prompt, str):
        append(UserMessage(content=task.data.prompt))
    else:
        for message in task.data.prompt or []:
            append(message)
    append(AssistantMessage(content=completion), sampled=True)
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        nodes=nodes,
    )


async def score_completion(task: vf.Task, trace: vf.Trace) -> float:
    await task.score(trace)
    if not trace.rewards or all(reward is None for reward in trace.rewards.values()):
        raise ValueError(f"task {type(task).__name__} produced no offline reward")
    trace.is_completed = True
    trace.ok = True
    return trace.reward


def _prompt_messages(task: vf.Task) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if task.data.system_prompt:
        messages.append({"role": "system", "content": task.data.system_prompt})
    if isinstance(task.data.prompt, str):
        messages.append({"role": "user", "content": task.data.prompt})
    else:
        messages.extend(
            message.model_dump(mode="json") if hasattr(message, "model_dump") else message
            for message in task.data.prompt or []
        )
    if not messages:
        raise ValueError("MPS mode currently requires a task with an opening prompt")
    return messages


def sample_completion(model, tokenizer, task: vf.Task, config: MPSConfig, device: torch.device) -> Rollout:
    prompt = tokenizer.apply_chat_template(
        _prompt_messages(task), tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(prompt, Mapping):
        prompt = prompt["input_ids"]
    if not isinstance(prompt, torch.Tensor):
        prompt = torch.tensor([prompt], dtype=torch.long)
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    prompt = prompt.to(device)

    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            prompt,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_completion_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    model.train(was_training)

    prompt_ids = prompt[0].tolist()
    completion_ids = generated[0, prompt.shape[1] :].tolist()
    if not completion_ids:
        raise ValueError("model generated no completion tokens")
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return Rollout(prompt_ids, completion_ids, completion, 0.0)


async def _score_rollouts(task: vf.Task, rollouts: list[Rollout]) -> None:
    rewards = await asyncio.gather(*(score_completion(task, make_trace(task, r.completion)) for r in rollouts))
    for rollout, reward in zip(rollouts, rewards):
        rollout.reward = reward


def _load_tasks(config: MPSConfig) -> list[vf.Task]:
    taskset_config = taskset_config_type(config.taskset)(id=config.taskset, **config.taskset_args)
    taskset = vf.load_taskset(taskset_config)
    if taskset.toolsets(taskset.config):
        raise ValueError("MPS mode currently supports tasksets without tools")
    tasks = list(itertools.islice(taskset, config.num_tasks))
    if not tasks:
        raise ValueError(f"taskset {config.taskset!r} produced no tasks")
    if any(type(task).toolsets(task.config) for task in tasks):
        raise ValueError("MPS mode currently supports tasks without tools")
    return tasks


def _load_policy(config: MPSConfig, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    if config.lora_rank:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules="all-linear",
            ),
        )
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def train(
    config: MPSConfig,
    *,
    model=None,
    tokenizer=None,
    tasks: list[vf.Task] | None = None,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    if device is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable; run this on an Apple Silicon Mac with MPS-enabled PyTorch")
        device = torch.device("mps")
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together")
    if model is None:
        model, tokenizer = _load_policy(config, device)
    else:
        model.to(device)
    tasks = tasks or _load_tasks(config)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("policy has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    task_cycle = itertools.cycle(tasks)
    metrics: list[dict[str, float]] = []
    model.train()

    for step in range(config.max_steps):
        task = next(task_cycle)
        rollouts = [sample_completion(model, tokenizer, task, config, device) for _ in range(config.group_size)]
        asyncio.run(_score_rollouts(task, rollouts))
        advantages = group_advantages([rollout.reward for rollout in rollouts])

        optimizer.zero_grad()
        total_loss = 0.0
        for rollout, advantage in zip(rollouts, advantages):
            sequence = torch.tensor([rollout.prompt_ids + rollout.completion_ids], dtype=torch.long, device=device)
            logits = model(sequence[:, :-1]).logits
            logprobs = completion_logprobs(logits, sequence, len(rollout.prompt_ids))
            loss = -advantage * logprobs.mean() / config.group_size
            loss.backward()
            total_loss += float(loss.detach())

        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        mean_reward = sum(rollout.reward for rollout in rollouts) / len(rollouts)
        row = {"step": float(step), "reward": mean_reward, "loss": total_loss}
        metrics.append(row)
        print(f"Step {step} | Reward {mean_reward:.4f} | Loss {total_loss:.4f}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        (config.output_dir / "mps_config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
    return metrics


def load_config(path: Path) -> MPSConfig:
    with path.open("rb") as file:
        return MPSConfig.model_validate(tomllib.load(file))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a group-relative policy-gradient experiment on Apple MPS")
    parser.add_argument("at_or_config", help="'@' or the TOML config path")
    parser.add_argument("config", nargs="?", help="TOML config path when the first argument is '@'")
    args = parser.parse_args(argv)
    path = args.config if args.at_or_config == "@" else args.at_or_config
    if path is None:
        parser.error("missing TOML config path")
    train(load_config(Path(path)))


if __name__ == "__main__":
    main()
