from types import SimpleNamespace

import prime_rl_mps.train as mps
import pytest
import torch
import verifiers.v1 as vf
from prime_rl_mps.train import (
    MPSConfig,
    Rollout,
    completion_logprobs,
    group_advantages,
    make_trace,
    score_completion,
    train,
)
from pydantic import ValidationError


class ExactTask(vf.Task):
    @vf.reward
    async def exact(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == "good")


class NoRewardTask(vf.Task):
    pass


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return torch.tensor([[0]])

    def decode(self, token_ids, **kwargs):
        return "good" if token_ids == [1] else "bad"

    def save_pretrained(self, output_dir):
        return None


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3))
        self.config = SimpleNamespace(use_cache=False)
        self.generation = 0

    def forward(self, input_ids):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=self.logits.expand(batch, length, -1))

    def generate(self, input_ids, **kwargs):
        self.generation += 1
        token = 1 if self.generation % 2 else 2
        return torch.cat([input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def save_pretrained(self, output_dir):
        return None


def test_config_is_strict_and_requires_a_group() -> None:
    with pytest.raises(ValidationError):
        MPSConfig(taskset="reverse-text", group_size=1)
    with pytest.raises(ValidationError):
        MPSConfig(taskset="reverse-text", unknown=True)


def test_group_advantages_normalize_and_collapse_equal_rewards() -> None:
    assert group_advantages([0.0, 1.0]) == pytest.approx([-1.0, 1.0])
    assert group_advantages([0.5, 0.5]) == [0.0, 0.0]


def test_completion_logprobs_select_only_generated_tokens() -> None:
    logits = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]])
    sequence = torch.tensor([[2, 2, 2, 1, 2]])
    actual = completion_logprobs(logits, sequence, prompt_length=3)
    expected = torch.log_softmax(logits.float(), dim=-1)[0, 2:4].gather(1, torch.tensor([[1], [2]])).squeeze(1)
    assert torch.allclose(actual, expected)


@pytest.mark.asyncio
async def test_trace_uses_verifiers_reward_hooks() -> None:
    task = ExactTask(vf.TaskData(idx=0, prompt="say good", system_prompt="system"))
    trace = make_trace(task, "good")
    assert [node.message.role for node in trace.nodes] == ["system", "user", "assistant"]
    assert await score_completion(task, trace) == 1.0
    no_reward = NoRewardTask(vf.TaskData(idx=1, prompt="hello"))
    with pytest.raises(ValueError, match="no offline reward"):
        await score_completion(no_reward, make_trace(no_reward, "answer"))


def test_one_training_step_updates_policy_on_cpu() -> None:
    model = TinyModel()
    before = model.logits.detach().clone()
    task = ExactTask(vf.TaskData(idx=0, prompt="say good"))
    config = MPSConfig(
        taskset="unused",
        max_steps=1,
        group_size=2,
        max_completion_tokens=1,
        output_dir=None,
        lora_rank=0,
    )

    metrics = train(
        config,
        model=model,
        tokenizer=TinyTokenizer(),
        tasks=[task],
        device=torch.device("cpu"),
    )

    assert metrics[0]["reward"] == pytest.approx(0.5)
    assert not torch.equal(model.logits.detach(), before)


def test_rollout_is_plain_training_data() -> None:
    rollout = Rollout(prompt_ids=[0], completion_ids=[1], completion="good", reward=1.0)
    assert rollout.completion_ids == [1]


def test_sample_completion_requires_prompt_and_tokens() -> None:
    config = MPSConfig(taskset="unused", max_completion_tokens=1, output_dir=None)
    with pytest.raises(ValueError, match="opening prompt"):
        mps.sample_completion(TinyModel(), TinyTokenizer(), ExactTask(vf.TaskData(idx=0)), config, torch.device("cpu"))


def test_sample_completion_accepts_transformers_batch_encoding() -> None:
    class BatchTokenizer(TinyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[0]])}

    config = MPSConfig(taskset="unused", max_completion_tokens=1, output_dir=None)
    rollout = mps.sample_completion(
        TinyModel(),
        BatchTokenizer(),
        ExactTask(vf.TaskData(idx=0, prompt="hello")),
        config,
        torch.device("cpu"),
    )
    assert rollout.prompt_ids == [0]

    class EmptyModel(TinyModel):
        def generate(self, input_ids, **kwargs):
            return input_ids

    with pytest.raises(ValueError, match="no completion tokens"):
        mps.sample_completion(
            EmptyModel(),
            TinyTokenizer(),
            ExactTask(vf.TaskData(idx=0, prompt="hello")),
            config,
            torch.device("cpu"),
        )


def test_task_loading_uses_typed_verifiers_config(monkeypatch) -> None:
    task = ExactTask(vf.TaskData(idx=0, prompt="hello"))

    class FakeTaskset:
        config = vf.TasksetConfig(id="fake")

        def toolsets(self, config):
            return []

        def __iter__(self):
            yield task

    monkeypatch.setattr(mps, "taskset_config_type", lambda taskset: vf.TasksetConfig)
    monkeypatch.setattr(vf, "load_taskset", lambda config: FakeTaskset())
    assert mps._load_tasks(MPSConfig(taskset="fake", num_tasks=1, output_dir=None)) == [task]


def test_policy_loader_uses_transformers_and_lora(monkeypatch) -> None:
    import peft
    import transformers

    model = TinyModel()
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(peft, "get_peft_model", lambda incoming, config: incoming)

    loaded_model, loaded_tokenizer = mps._load_policy(MPSConfig(taskset="unused", output_dir=None), torch.device("cpu"))
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert model.config.use_cache is False
    assert model.gradient_checkpointing


def test_train_validates_injected_policy_and_saves_config(tmp_path) -> None:
    config = MPSConfig(taskset="unused", max_steps=1, group_size=2, output_dir=tmp_path, lora_rank=0)
    task = ExactTask(vf.TaskData(idx=0, prompt="say good"))
    with pytest.raises(ValueError, match="provided together"):
        train(config, model=TinyModel(), tasks=[task], device=torch.device("cpu"))

    frozen = TinyModel()
    for parameter in frozen.parameters():
        parameter.requires_grad = False
    with pytest.raises(ValueError, match="no trainable parameters"):
        train(
            config,
            model=frozen,
            tokenizer=TinyTokenizer(),
            tasks=[task],
            device=torch.device("cpu"),
        )

    train(
        config,
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        tasks=[task],
        device=torch.device("cpu"),
    )
    assert (tmp_path / "mps_config.json").is_file()


def test_config_file_and_cli(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mps.toml"
    config_path.write_text('taskset = "reverse-text"\nmax_steps = 1\n')
    assert mps.load_config(config_path).taskset == "reverse-text"
    called = []
    monkeypatch.setattr(mps, "train", called.append)
    mps.main(["@", str(config_path)])
    assert called[0].taskset == "reverse-text"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_one_training_step_runs_on_mps() -> None:
    train(
        MPSConfig(
            taskset="unused",
            max_steps=1,
            group_size=2,
            max_completion_tokens=1,
            output_dir=None,
            lora_rank=0,
        ),
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        tasks=[ExactTask(vf.TaskData(idx=0, prompt="say good"))],
        device=torch.device("mps"),
    )
