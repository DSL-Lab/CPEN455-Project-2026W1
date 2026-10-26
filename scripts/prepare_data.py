"""Rebuild the public CBT splits in data/ from their pinned source. You do not need to run this:
the converted files ship in data/.

Children's Book Test (Hill et al., ICLR 2016), named-entity questions, from the Hugging Face
copy cam-cst/cbt (GNU FDL 1.3). Each item is 20 sentences of a children's book and a 21st
sentence with one named entity blanked out, plus 10 candidate entities from the passage. The
text is detokenised (the release is PTB-tokenised, which GPT-2 never saw), and items whose
context plus longest filled-in question exceed GPT-2's 1,024 tokens are dropped.

    validation (2,000 questions) -> data/cbt_ne_val.jsonl    real stories: check that your policy transfers
    test       (2,500 questions) -> data/cbt_ne_test.jsonl   more real stories (--test; 2,494 fit in 1,024 tokens)

LAMBADA (Paperno et al., ACL 2016), OpenAI version, from EleutherAI/lambada_openai (--lambada):
an open-vocabulary final word after a ~80-token passage (too short for the course budgets).

Reading the CBT parquet files needs pyarrow (not a course dependency).

    python scripts/prepare_data.py [--test] [--lambada]
"""
import argparse
import hashlib
import io
import json
import os
import urllib.request

import tiktoken

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CBT = dict(repo="cam-cst/cbt", revision="72b5c46b1248e3316360f0f2f0b2c39e773b68e4",
           files={"validation": ("NE/validation-00000-of-00001.parquet",
                                 "54ccd4a04d43a309ee6b0829ac737b7d7a70a0031827b58d31a4da080d07de97"),
                  "test": ("NE/test-00000-of-00001.parquet",
                           "670346965a8886436d9321d69aac10b36610a042a9d124390f88a258f5757af6")})
LAMBADA = dict(repo="EleutherAI/lambada_openai", revision="900124bf3b8235c6daf21033af9948b3f07346c4",
               path="data/lambada_test.jsonl", sha256="4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226")
MAX_LEN = 1024
BLANK = "XXXXX"


def fetch(repo, revision, path, sha256):
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
    print("downloading", url)
    raw = urllib.request.urlopen(url).read()
    assert hashlib.sha256(raw).hexdigest() == sha256, f"checksum mismatch for {url}"
    return raw


def detok(s):
    """Undo the PTB tokenisation of the CBT release ("son , I am" -> "son, I am")."""
    for a, b in ((" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), (" !", "!"), (" ?", "?"),
                 (" 's", "'s"), (" n't", "n't"), (" 're", "'re"), (" 've", "'ve"), (" 'll", "'ll"),
                 (" 'd", "'d"), (" 'm", "'m"), ("`` ", '"'), (" ''", '"'), ("``", '"'), ("''", '"'),
                 ("( ", "("), (" )", ")")):
        s = s.replace(a, b)
    return s


def fits(enc, context, question, fills):
    n = len(enc.encode(context))
    return n + max(len(enc.encode(" " + question.replace(BLANK, f))) for f in fills) <= MAX_LEN


def build_cbt(split, out, enc):
    import pyarrow.parquet as pq

    path, sha = CBT["files"][split]
    rows = pq.read_table(io.BytesIO(fetch(CBT["repo"], CBT["revision"], path, sha))).to_pylist()
    kept, dropped = [], 0
    for i, r in enumerate(rows):
        context = detok(" ".join(r["sentences"]))
        question = detok(r["question"])
        if not fits(enc, context, question, r["options"]):
            dropped += 1
            continue
        kept.append(dict(id=f"cbt-ne-{split}-{i:04d}", context=context,
                         questions=[dict(question=question, options=r["options"], answer=r["answer"], kind="NE")]))
    write(out, kept)
    print(f"{split}: {len(kept)} items kept, {dropped} longer than {MAX_LEN} tokens dropped -> {out}")


def build_lambada(out, enc):
    rows = [json.loads(l) for l in fetch(LAMBADA["repo"], LAMBADA["revision"], LAMBADA["path"], LAMBADA["sha256"]).decode().splitlines()]
    items = []
    for i, r in enumerate(rows):
        context, word = r["text"].rsplit(" ", 1)
        assert enc.encode(context) + enc.encode(" " + word) == enc.encode(r["text"])
        items.append(dict(id=f"lambada-{i:04d}", context=context,
                          questions=[dict(question=BLANK, options=None, answer=word, kind="final word")]))
    write(out, items)
    print(f"lambada: {len(items)} items -> {out}")


def write(path, items):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--test", action="store_true", help="also build the CBT test split (the submitted validation split)")
    ap.add_argument("--lambada", action="store_true", help="also build LAMBADA")
    args = ap.parse_args()
    enc = tiktoken.get_encoding("gpt2")
    build_cbt("validation", os.path.join(args.out, "cbt_ne_val.jsonl"), enc)
    if args.test:
        build_cbt("test", os.path.join(args.out, "cbt_ne_test.jsonl"), enc)
    if args.lambada:
        build_lambada(os.path.join(args.out, "lambada_test.jsonl"), enc)


if __name__ == "__main__":
    main()
