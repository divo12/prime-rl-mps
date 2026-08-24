import pytest
import torch
from prime_rl_mps.train import batched_completion_logprobs, grpo_losses, make_training_batch


def test_batched_logprobs_match_per_row_reference_gather() -> None:
    torch.manual_seed(0)
    rows, width = 3, 6
    logits = torch.randn(rows, width, 5)
    sequences = torch.randint(0, 5, (rows, width))
    prompt_lengths = [1, 3, 2]
    lengths = [width, 5, 4]
    actual = batched_completion_logprobs(logits, sequences, prompt_lengths, lengths)
    assert actual.shape == (rows, width - 1)
    for row in range(rows):
        start = prompt_lengths[row] - 1
        stop = lengths[row] - 1
        reference = torch.log_softmax(logits[row].float(), dim=-1)
        expected = reference[start:stop].gather(1, sequences[row, start + 1 : stop + 1, None]).squeeze(1)
        assert torch.allclose(actual[row, start:stop], expected)


def test_batch_masks_cover_exactly_the_completion_tokens() -> None:
    batch = make_training_batch(
        prompt_ids=[[1, 2], [3]],
        completion_ids=[[4, 5], [6]],
        pad_id=9,
        device=torch.device("cpu"),
    )
    assert batch.sequences.tolist() == [[1, 2, 4, 5], [3, 6, 9, 9]]
    assert batch.completion_mask.tolist() == [
        [False, True, True],
        [True, False, False],
    ]


def test_first_epoch_ratio_is_one_and_gradient_matches_plain_policy_gradient() -> None:
    advantages = torch.tensor([[1.0, -1.0], [2.0, 0.0]])
    mask = torch.ones(2, 2)
    new = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    old = new.clone()

    new.requires_grad_(True)
    loss, stats = grpo_losses(new, old.detach(), advantages, mask, epsilon=0.2)
    assert stats["mean_ratio"] == pytest.approx(1.0)
    assert stats["clip_fraction"] == pytest.approx(0.0)
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    expected_value = -(advantages * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected_value)

    loss.backward()
    new_plain = new.detach().clone().requires_grad_(True)
    plain = -(advantages * new_plain).sum() / mask.sum()
    plain.backward()
    assert torch.allclose(new.grad, new_plain.grad)


def test_clip_binds_above_for_positive_advantage_and_below_for_negative() -> None:
    mask = torch.ones(1, 1)

    new_high = torch.tensor([[5.0]])
    loss_up, stats_up = grpo_losses(new_high, torch.zeros(1, 1), torch.tensor([[1.0]]), mask, epsilon=0.2)
    assert stats_up["clip_fraction"] == pytest.approx(1.0)
    assert torch.allclose(loss_up, torch.tensor(-1.2))

    new_low = torch.tensor([[-5.0]])
    loss_down, stats_down = grpo_losses(new_low, torch.zeros(1, 1), torch.tensor([[-1.0]]), mask, epsilon=0.2)
    assert stats_down["clip_fraction"] == pytest.approx(1.0)
    assert torch.allclose(loss_down, torch.tensor(0.8))


def test_unclipped_side_of_objective_is_untouched_by_epsilon() -> None:
    mask = torch.ones(1, 1)
    new_small = torch.tensor([[-0.05]])
    old = torch.zeros(1, 1)
    advantages = torch.tensor([[1.0]])
    loss, stats = grpo_losses(new_small, old, advantages, mask, epsilon=0.2)
    ratio = torch.exp(new_small - old)
    assert torch.allclose(loss, -(ratio * advantages))
    assert stats["clip_fraction"] == pytest.approx(0.0)


def test_masked_positions_do_not_contribute_to_loss_or_stats() -> None:
    new = torch.tensor([[10.0, -0.1]])
    old = torch.tensor([[0.0, 0.0]])
    advantages = torch.tensor([[5.0, 5.0]])
    mask = torch.tensor([[False, True]])
    loss, stats = grpo_losses(new, old, advantages, mask, epsilon=0.2)
    assert torch.allclose(loss, -(torch.exp(torch.tensor(-0.1)) * 5.0))
    assert stats["clip_fraction"] == pytest.approx(0.0)
    assert stats["mean_ratio"] == pytest.approx(torch.exp(torch.tensor(-0.1)).item())
    assert stats["approx_kl"] != pytest.approx(0.0)
