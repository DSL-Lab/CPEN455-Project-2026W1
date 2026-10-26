"""Local quality Q, overhead factor R, Score and loss diagnostics.

Quality is measured only where eviction is forced. At budget B, a question whose context has
fewer than B + 2 tokens can be answered from the full context, so only the questions with
context length >= B + 2 are scored. The set depends on B alone, never on the policy, so every
policy is scored on the same questions (a policy with smaller layer capacities cannot add easier,
shorter contexts to its own set):

    Acc(B)   = accuracy of the policy on the scored questions
    Q(B)     = min(1, Acc(B) / Acc_full)        accuracy retained, on the same questions; 1 = lossless
    R(B)     = min(1, throughput_policy(B) / throughput_window(B))
    Score    = mean over budgets of Q(B) · R(B)

Loss differences before and after the first eviction are reported as diagnostics. These
measurements do not determine course marks; the leaderboard benchmark is TBA.
"""
import torch


def scored_from(policy):
    """Smallest context length of a scored question at this policy's budget (B + 2)."""
    return policy.budget + 2


def active_from(policy):
    """First absolute position whose prediction can be affected by an eviction."""
    caps = [policy.layer_capacity(l) for l in range(policy.n_layer)]
    if policy.kv_bits < 16:  # quantisation acts once tokens leave the fp16 residual window
        return policy.residual + 2
    return min(caps) + 2


def per_item(full, pol, start, q_start):
    """Per-item counts over the scored questions (context length >= q_start), and context-token
    loss changes before/after `start`."""
    out = dict(n_q=[], c_pol=[], c_full=[], d_ll=[], kinds={}, d_tok=0.0, n_tok=0,
               inactive_abs=0.0, inactive_max=0.0, n_inactive=0)
    for f, p in zip(full, pol):
        assert f["id"] == p["id"]
        act = f["ctx_len"] >= q_start
        qs = list(zip(f["questions"], p["questions"])) if act else []
        out["n_q"].append(len(qs))
        out["c_pol"].append(sum(qp["correct"] for _, qp in qs))
        out["c_full"].append(sum(qf["correct"] for qf, _ in qs))
        out["d_ll"].append(sum(qf["gold_ll"] - qp["gold_ll"] for qf, qp in qs))
        for qf, qp in qs:
            k = out["kinds"].setdefault(qf["kind"], [0, 0, 0])
            k[0] += 1
            k[1] += qp["correct"]
            k[2] += qf["correct"]
        d = p["token_nll"] - f["token_nll"]
        a = p["token_pos"] >= start
        out["d_tok"] += d[a].sum().item()
        out["n_tok"] += int(a.sum())
        if (~a).any():
            out["inactive_abs"] += d[~a].abs().sum().item()
            out["inactive_max"] = max(out["inactive_max"], d[~a].abs().max().item())
            out["n_inactive"] += int((~a).sum())
    return out


def quality_from(st, items=None):
    sel = (lambda xs: [xs[i] for i in items]) if items is not None else (lambda xs: xs)
    n, c_pol, c_full = sum(sel(st["n_q"])), sum(sel(st["c_pol"])), sum(sel(st["c_full"]))
    return dict(Q=min(1.0, c_pol / c_full) if c_full else 1.0, Acc=c_pol / n if n else None,
                Acc_full=c_full / n if n else None, n_active=n)


def quality(full, pol, start, q_start=None):
    """Q and diagnostics for one policy at one budget. `start` = active_from(policy),
    `q_start` = scored_from(policy) (defaults to `start`)."""
    q_start = start if q_start is None else q_start
    st = per_item(full, pol, start, q_start)
    q = quality_from(st)
    inactive_mean = st["inactive_abs"] / max(1, st["n_inactive"])
    q.update(active_from=start, scored_from=q_start, Q_ci95=bootstrap_q(st),
             dLL=sum(st["d_ll"]) / q["n_active"] if q["n_active"] else 0.0,
             dNLL=st["d_tok"] / st["n_tok"] if st["n_tok"] else 0.0,
             Acc_by_kind={k: dict(n=v[0], Acc=v[1] / v[0], Acc_full=v[2] / v[0]) for k, v in sorted(st["kinds"].items())},
             inactive_max_dnll=st["inactive_max"], inactive_mean_dnll=inactive_mean)
    return q


def bootstrap_q(st, n_boot=1000, seed=0, alpha=0.05):
    """95% percentile interval of Q, resampling items (stories) with replacement."""
    n = len(st["n_q"])
    g = torch.Generator().manual_seed(seed)
    qs = sorted(quality_from(st, torch.randint(n, (n,), generator=g).tolist())["Q"] for _ in range(n_boot))
    return qs[int(alpha / 2 * n_boot)], qs[int((1 - alpha / 2) * n_boot) - 1]


def overhead_factor(throughput, throughput_window):
    return min(1.0, throughput / throughput_window)


def score(per_budget):
    """per_budget: {budget: {"Q": .., "R": ..}} -> mean of Q·R."""
    return sum(v["Q"] * v["R"] for v in per_budget.values()) / len(per_budget)

