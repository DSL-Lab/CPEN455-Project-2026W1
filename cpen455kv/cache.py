"""The grader's KV cache: preallocated, budget-enforced, policy-driven.

This is the harness's own cache, used for every graded number. Students build a simpler
one themselves in Part 2 (student/kv_cache.py); that one is unit-tested, not used here.

Every sequence in a batch keeps its true absolute positions. Contexts are left-padded, and
padding slots carry position -1 and are never attended to. GPT-2 adds the position embedding
before K and V are computed, so a kept key needs no correction after other tokens are evicted.

The policy acts while the context streams in (`forward_cached`). The questions are then read on
top of the compressed cache without further eviction (`forward_frozen`, into `reserve` extra
slots per layer), and the cache is rolled back to the context after every option.
"""
import contextlib
import functools

import torch
import torch.nn.functional as F

from .flops import counter, non_sdpa_flops, sdpa_flops


class BudgetViolation(RuntimeError):
    pass


class BudgetedKVCache:
    def __init__(self, model_cfg, policy, batch, prefill_len, max_len, device, dtype, strict=True, reserve=0):
        self.policy, self.strict = policy, strict
        self.n_layer, self.H = model_cfg.n_layer, model_cfg.n_head
        self.D = model_cfg.n_embd // model_cfg.n_head
        self.B, self.device = batch, device
        if policy.capacity is None:
            self.caps = [None] * self.n_layer
        else:
            self.caps = [policy.layer_capacity(l) for l in range(self.n_layer)]
            if sum(self.caps) > self.n_layer * policy.capacity:
                raise BudgetViolation(f"layer capacities {self.caps} exceed {self.n_layer} x {policy.capacity}")
        size = [(max_len if c is None else min(max_len, max(prefill_len, c + 1))) + reserve for c in self.caps]
        self.k = [torch.empty(batch, self.H, s, self.D, device=device, dtype=dtype) for s in size]
        self.v = [torch.empty_like(t) for t in self.k]
        self.pos = [torch.full((batch, self.H, s), -1, device=device, dtype=torch.long) for s in size]
        self.n = [0] * self.n_layer
        # ring buffer of recent queries, only if the policy wants to probe attention
        self.W = policy.probe_window
        if self.W:
            self.q_hist = [torch.zeros(batch, self.H, self.W, self.D, device=device, dtype=dtype)
                           for _ in range(self.n_layer)]
            self.q_pos = [torch.full((batch, self.W), -1, device=device, dtype=torch.long)
                          for _ in range(self.n_layer)]
            self.q_ptr = [0] * self.n_layer
        self.count_policy_flops = False  # set by cpen455kv.timing to split model and policy costs
        self.policy_flops = 0            # FLOPs inside policy calls (excluding SDPA), if counted
        self.attn_flops = 0              # exact SDPA FLOPs of the model's attention calls

    # ---- storage -------------------------------------------------------------------------
    def view(self, l):
        n = self.n[l]
        return self.k[l][:, :, :n], self.v[l][:, :, :n], self.pos[l][:, :, :n]

    def append(self, l, k, v, pos_q):
        """k, v: [B, H, T, D]; pos_q: [B, T]. Returns the K, V, pos views including them."""
        n, T = self.n[l], k.shape[2]
        if n + T > self.k[l].shape[2]:
            raise BudgetViolation(f"layer {l}: {n} + {T} tokens exceed the allocated cache")
        self.k[l][:, :, n:n + T] = k
        self.v[l][:, :, n:n + T] = v
        self.pos[l][:, :, n:n + T] = pos_q[:, None, :]
        self.n[l] = n + T
        return self.view(l)

    def lengths(self):
        return list(self.n)

    def rollback(self, lengths):
        """Drop every slot appended after `lengths` (from `lengths()`), e.g. a scored question."""
        for l, n in enumerate(lengths):
            self.pos[l][:, :, n:self.n[l]] = -1
            self.n[l] = n

    def attn_mask(self, l, pos_q):
        """Boolean mask [B, H, T, L]: True where the query may attend to the key."""
        P = self.pos[l][:, :, :self.n[l]]
        valid = P >= 0
        T = pos_q.shape[1]
        if T == 1:  # one new token: every cached token is in its past
            return valid[:, :, None, :]
        m = valid[:, :, None, :] & (P[:, :, None, :] <= pos_q[:, None, :, None])
        # padding queries attend to their own slot so that no softmax row is empty
        ar = torch.arange(T, device=P.device)
        m[:, :, ar, self.n[l] - T + ar] = True
        return m

    # ---- attention probe -----------------------------------------------------------------
    def record_queries(self, l, q, pos_q):
        if not self.W:
            return
        t = min(q.shape[2], self.W)
        slots = (self.q_ptr[l] + torch.arange(t, device=q.device)) % self.W
        self.q_hist[l][:, :, slots] = q[:, :, -t:]
        self.q_pos[l][:, slots] = pos_q[:, -t:]
        self.q_ptr[l] = (self.q_ptr[l] + t) % self.W

    def probe(self, l, w=1):
        """softmax(q kᵀ/√D) of the last w queries against the current cache: [B, H, w, L]."""
        if not 1 <= w <= self.W:
            raise ValueError(f"probe({w}) needs 1 <= w <= policy.probe_window = {self.W}")
        slots = (self.q_ptr[l] - w + torch.arange(w, device=self.device)) % self.W
        Q, qp = self.q_hist[l][:, :, slots], self.q_pos[l][:, slots]
        K, _, P = self.view(l)
        s = (Q @ K.transpose(-1, -2)).float() * self.D ** -0.5
        ok = (P[:, :, None, :] >= 0) & (P[:, :, None, :] <= qp[:, None, :, None]) & (qp[:, None, :, None] >= 0)
        return torch.softmax(s.masked_fill(~ok, float("-inf")), dim=-1).nan_to_num(0.0)

    # ---- policy --------------------------------------------------------------------------
    def apply_policy(self, l, step, prefill):
        K, V, P = self.view(l)
        probe = functools.partial(self.probe, l)
        # a fresh counter per call: FlopCounterMode resets its counts on every __enter__
        fc = counter() if self.count_policy_flops else contextlib.nullcontext()
        with fc:
            if prefill:
                idx = self.policy.on_prefill(l, K, V, P, probe)
            else:
                idx = self.policy.on_step(l, K, V, P, step, probe)
            if idx is not None:
                self._keep(l, idx)
            elif self.caps[l] is not None and self.n[l] > self.caps[l]:
                raise BudgetViolation(f"layer {l}: {self.n[l]} tokens > capacity {self.caps[l]}, no indices returned")
            if self.policy.kv_bits < 16:
                self.policy.compress(l, *self.view(l))  # modifies the cache views in place
        if self.count_policy_flops:
            self.policy_flops += non_sdpa_flops(fc)

    def _keep(self, l, idx):
        n, cap = self.n[l], self.caps[l]
        if idx.dim() != 3 or tuple(idx.shape[:2]) != (self.B, self.H):
            raise BudgetViolation(f"layer {l}: indices must be [B, H, M], got {tuple(idx.shape)}")
        m = idx.shape[-1]
        if cap is not None and m > cap:
            raise BudgetViolation(f"layer {l}: kept {m} tokens > capacity {cap}")
        if cap is not None and n <= cap and self.policy.kv_bits >= 16:
            # quality is scored only after the policy kicks in, so it may not act before then
            raise BudgetViolation(f"layer {l}: indices returned while the cache ({n}) was within capacity {cap}")
        idx = idx.to(device=self.device, dtype=torch.long)
        if self.strict:
            s = idx.sort(dim=-1).values
            if s.numel() and (s[..., 0].min() < 0 or s[..., -1].max() >= n or (s.diff(dim=-1) == 0).any()):
                raise BudgetViolation(f"layer {l}: indices must be distinct and in [0, {n})")
        # index_select on the flattened buffer: much faster than gather with an expanded
        # index on some backends (MPS), and it reads only the kept rows
        S = self.k[l].shape[2]
        rows = (idx + (torch.arange(self.B * self.H, device=self.device) * S).view(self.B, self.H, 1)).flatten()
        for buf in (self.k[l], self.v[l]):
            buf[:, :, :m] = buf.view(-1, self.D).index_select(0, rows).view(self.B, self.H, m, self.D)
        self.pos[l][:, :, :m] = self.pos[l].view(-1).index_select(0, rows).view(self.B, self.H, m)
        self.pos[l][:, :, m:n] = -1
        self.n[l] = m


def forward_cached(model, cache, idx, pos_q, step, prefill):
    """One forward pass of nanoGPT through the budgeted cache.

    idx, pos_q: [B, T] token ids and absolute positions (-1 = padding).
    Returns the logits of the last position, [B, vocab]."""
    tr, cfg = model.transformer, model.config
    B, T = idx.shape
    H, C = cfg.n_head, cfg.n_embd
    x = tr.wte(idx) + tr.wpe(pos_q.clamp(0, cfg.block_size - 1))
    for l, blk in enumerate(tr.h):
        q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        K, V, _ = cache.append(l, k, v, pos_q)
        cache.attn_flops += sdpa_flops(B, H, T, K.shape[2], C // H)
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=cache.attn_mask(l, pos_q))
        x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + blk.mlp(blk.ln_2(x))
        cache.record_queries(l, q, pos_q)
        cache.apply_policy(l, step, prefill)
    return model.lm_head(tr.ln_f(x[:, -1]))


def forward_frozen(model, cache, idx, pos_q):
    """A forward pass that appends to the cache but never calls the policy (reading a question
    on top of the compressed context). Returns the logits of every position, [B, T, vocab]."""
    tr, cfg = model.transformer, model.config
    B, T = idx.shape
    H, C = cfg.n_head, cfg.n_embd
    x = tr.wte(idx) + tr.wpe(pos_q.clamp(0, cfg.block_size - 1))
    for l, blk in enumerate(tr.h):
        q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        K, V, _ = cache.append(l, k, v, pos_q)
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=cache.attn_mask(l, pos_q))
        x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + blk.mlp(blk.ln_2(x))
    return model.lm_head(tr.ln_f(x))
