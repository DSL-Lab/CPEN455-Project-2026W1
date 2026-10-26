"""Reading tasks: loading, GPT-2 tokenisation, batching.

Every split is a jsonl file of items. An item is a context (a story) and one or more questions
about it:

    {"id": "cbt-ne-validation-0000", "context": "... Johnny Chuck had escaped from old Whitetail ...",
     "questions": [{"question": "Johnny XXXXX looked up to see a dog almost upon her.",
                    "options": ["Chuck", "Johnny", "Marshhawk", ...], "answer": "Chuck", "kind": "NE"}]}

A question is a sentence with one blank (XXXXX). The harness streams the context through the
cache, then scores every option by the log-likelihood of the question sentence with the blank
filled in, on top of the compressed cache, and picks the most likely option (the GPT-2 paper's
protocol for the Children's Book Test). `options: null` makes an open-vocabulary question: it
is correct only if every token of the answer is the model's top-1 prediction (LAMBADA).

Each loaded question also carries `distance`: the number of context tokens after the last
mention of the answer (None if the answer never appears in the context).
"""
import functools
import json
import os
import random
import re

import tiktoken
import torch

from .env import data_dir

EOT = 50256
BLANK = "XXXXX"
MAX_LEN = 1024  # GPT-2's context: the context plus the longest filled-in question must fit
SPLIT_FILES = {
    "cbt-val": "cbt_ne_val.jsonl",    # Children's Book Test (named entities): development and tuning
    "cbt-test": "cbt_ne_test.jsonl",  # Children's Book Test (named entities): reported validation, validation.json
    "lambada": "lambada_test.jsonl",  # LAMBADA (OpenAI version), open-vocabulary final word
}


@functools.lru_cache(maxsize=None)
def encoder():
    return tiktoken.get_encoding("gpt2")


def encode_question(q):
    """Token ids of the question sentence, once per option (open questions: the answer only)."""
    enc = encoder()
    assert q["question"].count(BLANK) == 1, q["question"]
    fills = q["options"] if q.get("options") is not None else [q["answer"]]
    return [enc.encode(" " + q["question"].replace(BLANK, f)) for f in fills]


def answer_distance(ctx, answer):
    """Tokens after the last exact, case-sensitive, whole-word mention of `answer`.

    Count original context tokens, excluding any token overlapping the mention.
    Return None when the answer is empty or absent; do not normalize either text.
    """
    enc = encoder()
    text = enc.decode(ctx)
    matches = list(re.finditer(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text)) if answer else []
    if not matches:
        return None
    # Match complete entities (Nag must not match Nagaina), then map the end of the
    # mention to the original token boundaries, including multi-byte characters.
    end_byte = len(text[:matches[-1].end()].encode("utf-8"))
    offset = 0
    for i, token in enumerate(ctx):
        offset += len(enc.decode_single_token_bytes(token))
        if offset >= end_byte:
            return len(ctx) - i - 1
    return None


def load_item(r):
    enc = encoder()
    ctx = enc.encode(r["context"])
    questions = []
    for q in r["questions"]:
        opts = encode_question(q)
        assert len(ctx) + max(map(len, opts)) <= MAX_LEN, f"{r['id']} does not fit in GPT-2's context"
        is_open = q.get("options") is None
        questions.append(dict(options=opts, answer=0 if is_open else q["options"].index(q["answer"]),
                              open=is_open, kind=q.get("kind", ""),
                              distance=q["distance"] if "distance" in q else answer_distance(ctx, q["answer"])))
    return dict(id=r["id"], ctx=ctx, questions=questions)


def load_split(split, limit=None, seed=None):
    """Return a list of items with token ids.

    `split` is a name in SPLIT_FILES or a path to a jsonl file in the same format.
    `seed` shuffles before `limit` is applied, for reproducible subsets."""
    path = split if split.endswith(".jsonl") else os.path.join(data_dir(), SPLIT_FILES[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found (benchmark splits: python scripts/prepare_data.py)")
    rows = [json.loads(line) for line in open(path)]
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return [load_item(r) for r in rows]


def make_batches(items, batch_size):
    """Group items of similar context length so that left-padding is small."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ctx"]))
    return [[items[i] for i in order[j:j + batch_size]] for j in range(0, len(order), batch_size)]


def collate(batch, device):
    """Left-pad the contexts so that they all end at the last column.

    Returns context ids [B, P] and positions [B, P] (-1 at padding)."""
    B = len(batch)
    P = max(len(it["ctx"]) for it in batch)
    ids = torch.full((B, P), EOT, dtype=torch.long)
    pos = torch.full((B, P), -1, dtype=torch.long)
    for b, it in enumerate(batch):
        n = len(it["ctx"])
        ids[b, P - n:] = torch.tensor(it["ctx"])
        pos[b, P - n:] = torch.arange(n)
    return ids.to(device), pos.to(device)
