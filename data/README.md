# Data

Every file is a jsonl of items `{"id", "context", "questions": [...]}`; each question has a
`question` sentence with one `XXXXX` blank, ten `options`, the `answer` and its `kind`. See the
README, section *Dataset*, and `cpen455kv/data.py`.

| File | `--split` | Items | Questions | Source |
|---|---|---|---|---|
| `cbt_ne_val.jsonl` | `cbt-val` | 2,000 | 2,000 | Children's Book Test, named entities, validation split (`python scripts/prepare_data.py`) |
| `cbt_ne_test.jsonl` | `cbt-test` | 2,494 | 2,494 | Children's Book Test, named entities, test split (`python scripts/prepare_data.py --test`; 6 of its 2,500 items exceed GPT-2's 1,024 tokens and are dropped) |

Use `cbt-val` for development and tuning, and `cbt-test` for the reported validation run
(`validation.json`). `SHA256SUMS` pins the files. Report results on both splits; keep all tuning
and training decisions on `cbt-val`.

When a question is loaded, `cpen455kv/data.py` adds `distance`: the number of context tokens after
the end of the last complete mention of the answer, so `Nag` does not match `Nagaina`.
The count uses the existing context tokenization; a token overlapping the mention is excluded.
An answer ending at the context's end has distance 0; an absent answer has distance `null`.
Use it for the accuracy-vs-distance analysis in F1, reporting absent mentions separately.

**Children's Book Test** (Hill, Bordes, Chopra, Weston. *The Goldilocks Principle: Reading
Children's Books with Explicit Memory Representations.* ICLR 2016, arXiv:1511.02301), from the
Hugging Face copy `cam-cst/cbt` at revision `72b5c46b1248e3316360f0f2f0b2c39e773b68e4`. Each item
is 20 consecutive sentences of a children's book from Project Gutenberg and the 21st sentence with
one named entity blanked out, with ten candidates from the passage. The text is detokenised
(`scripts/prepare_data.py`). Distributed under the GNU Free Documentation License 1.3.
