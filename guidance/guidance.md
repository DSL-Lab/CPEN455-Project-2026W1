# Project workflow

The [README](../README.md) is the handout and includes the overall grade breakdown. The [grading rubric](grading.md) gives point allocations; use the [report checklist](../report/README.md) for required experiments and questions.

## Setup

Use the uv or Conda commands in the [README](../README.md#getting-started). Both install Python 3.11 and the same dependencies from `uv.lock`: CUDA 12.6 on NVIDIA machines, or the CPU/MPS variant for laptops. The CUDA build requires a compatible NVIDIA driver; it does not require students to build CUDA libraries.

Activate the environment, then `source scripts/env.sh` before starting Python. For uv, activation is `source .venv/bin/activate`; for Conda it is `conda activate cpen455`. On notebooks use `.venv/bin/python` explicitly, since activation does not persist between cells.

`python -m cpen455kv.env --strict` checks direct package pins, Python version and runtime settings. Add `--json environment.json` to save the software, hardware and checkpoint metadata; `grade` and `part2_bench` include it automatically. GPU runs use fp16 by default and CPU runs use fp32. Record your actual device and dtype.

`pyproject.toml` defines dependencies, `uv.lock` fixes their resolution, and `requirements*.txt` are generated Conda/pip exports. Do not change these files in your submission. Do not install inference engines or cache libraries. To repeat an install without downloading model weights, run `bash scripts/setup_t4.sh --no-download` (add `--cpu` for a laptop).

## Suggested order

1. Read `nanogpt/model.py` and the contracts in `student/kv_cache.py`. Complete the concept check.
2. Implement the cache TODOs, investigate the provided code and add regression tests in `student/my_tests/`. Keep `student/MUTANTS.md` up to date.
3. Read `cpen455kv/policy_base.py` and `cpen455kv/cache.py`, then implement H2O.
4. Develop and tune MyPolicy on `cbt-val` only; keep configuration changes in `student/config.yaml`. Treat `cbt-test` as held out.
5. Freeze your method and run validation, required experiments and the report checklist.

Public tests do not cover every requirement. Preserve function interfaces and justify your implementation from the model and API contracts. Do not change the harness or public tests in your submission.

## Useful commands

Run these commands from the project root in the activated environment.

```bash
python -m pytest tests/test_kv_cache.py -q
python -m pytest tests/test_policies.py -q
python -m pytest student/my_tests -q
python -m cpen455kv.grade --split cbt-val --limit 64 --skip-timing
CPEN455_POLICY=H2O python -m cpen455kv.grade --split cbt-val --out h2o_cbt_val.json
python -m cpen455kv.grade --strict-env --split cbt-test --out validation.json
python -m cpen455kv.part2_bench --strict-env --out student/part2_results.json
python -m cpen455kv.check .
```

The local reference policies come from your `student/policy.py`; incomplete H2O or baseline bugs can affect comparisons. Quality-only runs omit measured efficiency. Record the checkpoint, GPU, dtype, arguments and item IDs for each experiment.

`grade` writes per-budget quality and timing results. `harness.evaluate_batch` exposes per-question predictions and option scores, plus context-token losses. Use these interfaces for your analysis; the [report checklist](../report/README.md#experiment-protocols) specifies the comparisons.

For final submission, run the complete `cbt-test` split with `--strict-env` in the locked environment, check the [report checklist](../report/README.md), then build your ZIP and run `python -m cpen455kv.check submission.zip` as described in the [README](../README.md#submission-requirements). Bonus code, training and weights are a separate `bonus/` submission; they do not replace the core checkpoint or harness.

## Concept check

Answer in `student/part1_answers.yaml` with short explanations. Here L is context length, B is batch size and d is model width.

1. Give the dimensions and dimension names of one GPT-2-medium layer's key cache after 300 tokens at batch size 8.
2. Derive total KV-cache memory and evaluate it for GPT-2-medium at L = 1,024, B = 64 and fp16. Find the cache token budget whose memory equals vanilla generation's approximate `12·B·L·d·2` activation bytes. Give integer byte counts and explain units.
3. Why cache keys and values, but not queries?
4. Derive the leading-order total attention FLOPs for generating T tokens with and without a cache. State assumptions and exclude non-attention operations.

Be ready to explain selected lines of your submitted MyPolicy after the deadline.

## Bonus experiments

Choose one of the [three bonus directions](../README.md#bonus-points). Use GPT-2 small or medium and identify the checkpoint; keep core results on unchanged GPT-2-medium.

- For training, sort the 2,000 `cbt-val` items by ID: use the first 1,600 for fitting and the last 400 for model selection. Fit normalisation and training targets on the fitting subset only. Use training seeds 0, 1 and 2. Freeze choices before evaluating on `cbt-test`.
- Evaluate on the first 128 eligible `cbt-test` items in file order, or a larger fixed subset you identify. For generation, require at least 128 context tokens, use the first 128 as the prompt, and generate 256 tokens without stopping at EOS. For cloze evaluation, use complete contexts and the core scored-question rule.
- Compare methods on the same checkpoint, prompts, device, dtype and batch size. Report three timing repeats (median and range), peak memory, and 95% item-bootstrap intervals for quality/fidelity means (1,000 resamples, seed 0). Show individual training-seed results and their mean where applicable.
- For MTP, compare ordinary decoding with the same target weights and include drafting, verification and rejection costs. Use full caches for both. If the backbone is fine-tuned, also compare prediction quality with the original checkpoint; report numerical disagreements honestly.
- For compression fidelity, use a 16-token prefill, then ingest the rest of the prompt one token at a time with the policy active. Use the same schedule for full cache. Check the measurement with full-vs-full generation and a known deliberate token mismatch.
- Include a controlled ablation, interpretation and limitations. Supply commands, code/config, subset IDs, data hashes, seeds, trained state dictionaries where applicable, and raw results in `bonus/`.
