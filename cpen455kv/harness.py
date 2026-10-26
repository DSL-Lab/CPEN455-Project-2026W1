"""Quality evaluation of a cache policy on reading tasks.

For each item the harness prefills the first PREFILL context tokens with the full cache, then
streams the rest of the context one token at a time through the cached model, letting the
policy evict after every token. Then it reads each question on top of the compressed cache,
once per option and without further eviction, and scores the option by the log-likelihood of
the question sentence with the blank filled in; the most likely option is the model's answer
(see cpen455kv.data). Nothing is sampled, so the result is deterministic.

Along the way it records the loss of every streamed context token, a diagnostic of how well the
compressed model still predicts the story itself.
"""
import time

import torch

from .cache import BudgetedKVCache, forward_cached, forward_frozen
from .data import EOT, collate, make_batches

PREFILL = 16  # context tokens prefilled at once; every later token is streamed


def _score_tails(model, cache, last_logp, tails, n_ctx, device):
    """Log-likelihood and greedy exact match of one token sequence per row on top of `cache`.

    tails: list of B token lists (None = nothing to score in this row). The first token of a
    tail is predicted by `last_logp` [B, vocab], the output of the last context token."""
    B = len(tails)
    T = max(len(t) for t in tails if t)
    tail = torch.full((B, T), EOT, dtype=torch.long)
    lens = torch.ones(B, dtype=torch.long)
    for b, t in enumerate(tails):
        if t:
            tail[b, :len(t)] = torch.tensor(t)
            lens[b] = len(t)
    tail, lens = tail.to(device), lens.to(device)
    base = cache.lengths()
    logits = forward_frozen(model, cache, tail, n_ctx[:, None] + torch.arange(T, device=device)[None]).float()
    cache.rollback(base)
    lp = torch.cat([last_logp.gather(1, tail[:, :1]),
                    logits[:, :-1].gather(2, tail[:, 1:, None])[..., 0] - logits[:, :-1].logsumexp(-1)], dim=1)
    top = torch.cat([last_logp.argmax(-1, keepdim=True), logits[:, :-1].argmax(-1)], dim=1)
    valid = torch.arange(T, device=device)[None] < lens[:, None]
    return (lp * valid).sum(1).tolist(), ((top == tail) | ~valid).all(1).tolist()


@torch.no_grad()
def evaluate_batch(model, batch, policy, device, dtype, strict=True):
    """Stream the contexts of `batch` through `policy`, then answer every question.

    Returns one dict per item: ctx_len, the positions and losses of the streamed context tokens,
    and per question the option log-likelihoods, the prediction and whether it is correct."""
    ids, pos = collate(batch, device)
    B, P = ids.shape
    p0 = min(PREFILL, P)
    reserve = max(len(o) for it in batch for q in it["questions"] for o in q["options"])
    policy.reset(B, device)
    cache = BudgetedKVCache(model.config, policy, B, p0, P, device, dtype, strict, reserve=reserve)

    nll = torch.zeros(B, P, device=device)  # nll[:, s]: loss of context token s, predicted at step s - 1
    logits = forward_cached(model, cache, ids[:, :p0], pos[:, :p0], step=0, prefill=True)
    for s in range(p0, P):
        nll[:, s] = -torch.log_softmax(logits.float(), dim=-1).gather(1, ids[:, s:s + 1])[:, 0]
        logits = forward_cached(model, cache, ids[:, s:s + 1], pos[:, s:s + 1], step=s - p0 + 1, prefill=False)
    last_logp = torch.log_softmax(logits.float(), dim=-1)
    n_ctx = (pos >= 0).sum(1)

    scores = [[[] for _ in it["questions"]] for it in batch]
    greedy = [[False] * len(it["questions"]) for it in batch]
    for qi in range(max(len(it["questions"]) for it in batch)):
        n_opt = max(len(it["questions"][qi]["options"]) for it in batch if qi < len(it["questions"]))
        for oi in range(n_opt):
            tails = [it["questions"][qi]["options"][oi]
                     if qi < len(it["questions"]) and oi < len(it["questions"][qi]["options"]) else None
                     for it in batch]
            ll, em = _score_tails(model, cache, last_logp, tails, n_ctx, device)
            for b, t in enumerate(tails):
                if t:
                    scores[b][qi].append(ll[b])
                    greedy[b][qi] = em[b]

    scored = (pos >= 1) & (torch.arange(P, device=device)[None] >= p0)  # streamed, with a real token before
    out = []
    for b, it in enumerate(batch):
        qs = []
        for qi, q in enumerate(it["questions"]):
            pred = max(range(len(scores[b][qi])), key=scores[b][qi].__getitem__)
            correct = greedy[b][qi] if q["open"] else pred == q["answer"]
            qs.append(dict(scores=scores[b][qi], pred=pred, correct=bool(correct), gold_ll=scores[b][qi][q["answer"]],
                           gold_len=len(q["options"][q["answer"]]), kind=q["kind"], distance=q["distance"]))
        m = scored[b]
        out.append(dict(id=it["id"], ctx_len=len(it["ctx"]), questions=qs,
                        token_pos=pos[b][m].cpu(), token_nll=nll[b][m].cpu()))
    return out


def evaluate_quality(model, items, make_policy, batch_size, device, dtype, strict=True, verbose=False):
    """Run `evaluate_batch` over `items`; returns one dict per item, in the order of `items`."""
    by_id = {}
    t0 = time.time()
    batches = make_batches(items, batch_size)
    for i, batch in enumerate(batches):
        for r in evaluate_batch(model, batch, make_policy(), device, dtype, strict):
            by_id[r["id"]] = r
        if verbose:
            print(f"    batch {i + 1}/{len(batches)}  {time.time() - t0:.0f}s", end="\r", flush=True)
    if verbose:
        print()
    return [by_id[it["id"]] for it in items]
