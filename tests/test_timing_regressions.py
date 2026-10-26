"""FLOPs must describe the same decoding workload as the throughput measurement."""

import pytest
import torch

from cpen455kv import timing
from cpen455kv.policy_base import FullCache, Window
from nanogpt.model import GPT, GPTConfig


class PeriodicWorkWindow(Window):
    """Add a fixed synthetic workload to the provided sliding-window policy."""

    def reset(self, batch_size, device):
        self.work_left = torch.ones(batch_size, 7, 11, device=device)
        self.work_right = torch.ones(batch_size, 11, 5, device=device)

    def on_step(self, layer, k, v, pos, step, probe):
        if step > 0 and (step - 1) % 32 == 0:
            torch.bmm(self.work_left, self.work_right)
        return super().on_step(layer, k, v, pos, step, probe)


@pytest.mark.parametrize("warmup,steps,workloads", [
    (0, 31, 1),
    (0, 32, 1),
    (0, 33, 2),
    (1, 31, 0),
    (1, 32, 1),
    (1, 33, 1),
])
def test_interval32_flops_cover_only_the_timed_steps(warmup, steps, workloads):
    """Charge partial cycles correctly, excluding periodic work done during warmup."""
    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=16, vocab_size=64,
                    block_size=128, bias=True, dropout=0.0)
    model = GPT(cfg).eval()
    result = timing.decode_benchmark(
        model, [{"ctx": [1] * 128}], lambda: PeriodicWorkWindow(32, cfg),
        2, "cpu", torch.float32, prompt_len=32, warmup=warmup, steps=steps, repeats=1,
    )
    # Synthetic matrix multiplies occur at decoding steps 1, 33, 65, ... .
    per_workload = 2 * cfg.n_layer * 7 * 11 * 5
    expected = workloads * per_workload / steps / 1e9
    assert result["policy_gflops_per_token"] == pytest.approx(expected, rel=1e-6, abs=1e-15)
    assert result["model_gflops_per_token"] == pytest.approx(result["analytic_gflops_per_token"], rel=1e-6)


def test_flop_replay_matches_timed_positions_without_instrumenting_latency(monkeypatch):
    """The last legal context position needs no extra tail reserved for FLOP samples."""
    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=16, vocab_size=64,
                    block_size=37, bias=True, dropout=0.0)
    model = GPT(cfg).eval()
    calls = {}
    real_forward = timing.forward_cached

    def record_forward(model, cache, idx, pos_q, step, prefill):
        calls.setdefault(cache, []).append((tuple(pos_q[0].tolist()), step, prefill, cache.count_policy_flops))
        return real_forward(model, cache, idx, pos_q, step, prefill)

    monkeypatch.setattr(timing, "forward_cached", record_forward)
    result = timing.decode_benchmark(
        model, [{"ctx": [1] * 37}], lambda: FullCache(None, cfg),
        1, "cpu", torch.float32, prompt_len=4, warmup=2, steps=31, repeats=2,
    )

    first, second, replay = calls.values()
    assert first == second
    assert all(not counting for _, _, _, counting in first)
    assert [entry[:3] for entry in replay] == [entry[:3] for entry in first]
    assert all(not counting for _, _, _, counting in replay[:3])  # prefill + warmup
    assert all(counting for _, _, _, counting in replay[3:])
    assert replay[3][:2] == ((6,), 3)
    assert replay[-1][:2] == ((36,), 33)
    assert len(replay[3:]) == 31
    assert len(result["times_s"]) == 2
