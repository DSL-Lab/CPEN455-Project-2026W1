# CPEN 455 2026 W1 Course Project: Cache on a Budget

This is the course project for CPEN 455 at UBC, 2026 Winter Term 1. You will **build a KV cache for nanoGPT, implement the H2O eviction policy, and design a policy of your own that improves on it.** We evaluate how much reading-task accuracy your policy preserves under a fixed memory budget, counting the cost of deciding what to keep.

## Contents

- [Project introduction](#project-introduction)
- [Getting started](#getting-started)
- [Dataset](#dataset)
- [Project tasks](#project-tasks)
- [Model evaluation](#model-evaluation)
- [Grading rubric](#grading-rubric)
- [Bonus points](#bonus-points)
- [Submission requirements](#submission-requirements)
- [Report and milestones](#report-and-milestones)
- [Academic integrity](#academic-integrity)

## Project introduction

A decoder-only transformer generates text one token at a time. At decoding step $t$, each attention head compares the new token's query $q_t$ with the keys of all tokens so far, $K_{\le t}$, and returns a weighted sum of their values, $V_{\le t}$:

$$
\mathrm{Attn}(q_t, K_{\le t}, V_{\le t}) = \mathrm{softmax}\left( \frac{q_t K_{\le t}^{\top}}{\sqrt{d_h}} \right) V_{\le t}
$$

where $d_h$ is the dimension of each attention head.

Without caching, every decoding step recomputes the keys and values of the entire prefix. Most of that work is redundant. Attention is causal, so a token's keys and values depend only on tokens up to and including that token; with fixed weights and an unchanged prefix, they never change once computed. The **KV cache** stores these tensors at every layer. The model processes the prompt once, and each later step computes keys and values only for the new token, which then attends to everything stored.

The cache trades computation for memory, and that memory grows with context length. Under a fixed memory budget, the cache must eventually drop tokens, so the real question is *which tokens to keep*. In this project you will implement the cache, fix bugs deliberately planted in the starter code, and design an eviction policy that answers that question well.

The core project uses **GPT-2-medium with its original, unchanged weights**, running in nanoGPT; GPT-2 small is available for faster development. GPT-2 uses learned absolute position embeddings and has a 1,024-token context limit. The core project requires no training; training appears only in the optional [bonus](#bonus-points).

| Location | Purpose |
|---|---|
| `student/kv_cache.py` | Cache implementation and generation |
| `student/policy.py`, `student/config.yaml` | H2O and your improved `MyPolicy` |
| `student/part1_answers.yaml`, `student/MUTANTS.md`, `student/my_tests/` | Written answers, bug log and your own tests |
| `cpen455kv/`, `nanogpt/`, `tests/` | Provided harness, model backbone and public tests (do not modify these in your submission) |
| `data/`, `scripts/`, `notebooks/` | Public data and setup tools |
| [Guidance](guidance/guidance.md), [grading rubric](guidance/grading.md), [report checklist](report/README.md) | Workflow, point allocations, experiment protocols and report deliverables |

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the following from the project root. Supported platforms are Linux or WSL on x86-64, and Apple-silicon macOS.

```bash
bash scripts/setup_t4.sh             # add --cpu on a CPU-only or Apple-silicon laptop
source .venv/bin/activate
source scripts/env.sh
python -m cpen455kv.env --strict
python -m pytest tests -q
```

The setup script creates a Python 3.11 environment from `uv.lock` and downloads both base checkpoints (GPT-2 small and GPT-2-medium). The [T4 notebook](notebooks/t4_quickstart.ipynb) performs the same setup on a hosted T4 GPU. After setup, the harness tests should pass; the starter tests are expected to fail until you complete the TODOs and fix the planted bugs.

<details>
<summary>Conda alternative</summary>

```bash
conda create -n cpen455 python=3.11 pip -y
conda activate cpen455
python -m pip install --require-hashes -r requirements.txt
python scripts/download_gpt2.py gpt2
python scripts/download_gpt2.py gpt2-medium
source scripts/env.sh
python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
python -m cpen455kv.env --strict
```

On a CPU-only machine or Apple silicon, install `requirements-cpu.txt` instead. Both requirements files are generated from the same `uv.lock`; do not edit them or install other package versions.
</details>

Every time you open a new shell, activate your environment (uv or Conda) and run `source scripts/env.sh` **before** starting Python. The commands you will use most often are:

```bash
# Quick development checks
python -m pytest student/my_tests -q
python -m cpen455kv.grade --split cbt-val --limit 64 --skip-timing

# Final measurements: F1 on the development split, the submitted validation run, Part 2
python -m cpen455kv.grade --strict-env --split cbt-val --out cbt_val.json
python -m cpen455kv.grade --strict-env --split cbt-test --out validation.json
python -m cpen455kv.part2_bench --strict-env --out student/part2_results.json
```

**Hardware.** Use the device available to you and record its model and dtype in your report. Keep the same device, dtype and settings across compared methods. Use only the course dependencies; do not add inference engines or KV-cache libraries. See the [guidance](guidance/guidance.md) for setup checks.

## Dataset

Both public splits come from the Children's Book Test (CBT). Each item is a context (20 consecutive sentences of a children's book) followed by a cloze question: the 21st sentence with one named entity replaced by a blank, `XXXXX`, and ten candidate answers. After reading the context through your cache, the harness fills the blank with each candidate and picks the one whose sentence has the highest log-likelihood.

| Split | Items | Use |
|---|---:|---|
| `cbt-val` | 2,000 CBT stories | Development, tuning and any bonus training |
| `cbt-test` | 2,494 CBT stories | Reported validation (`validation.json`); never tune on it |

Make every design and tuning decision on `cbt-val`, and treat `cbt-test` as held out: run it to report results, never to choose settings or to train. Learned bonus components use separate fitting and model-selection subsets of `cbt-val`; see the [bonus protocol](guidance/guidance.md#bonus-experiments).

The data format, provenance and checksums are documented in [data/README.md](data/README.md). Performance will also be evaluated on held-out data.

## Project tasks

1. **Concept check.** Answer the four [concept questions](guidance/guidance.md#concept-check) in `student/part1_answers.yaml`.
2. **Build the cache.** Implement `generate_with_cache`, `KVCache.evict_to` and `generate_window`. Check that full-cache logits and greedy output match vanilla nanoGPT. Then use `cpen455kv.part2_bench` to measure vanilla, full-cache and 256-token-window generation.
3. **Mutant hunt.** The provided (non-TODO) code in `student/kv_cache.py` and `student/policy.py` contains nine deliberate bugs. Find and fix each one, and record its cause, fix and regression test in `student/MUTANTS.md`. The public tests catch only some of them.
4. **Implement H2O.** Follow the function contracts in `student/policy.py`, including scoring, protected tokens and eviction intervals. The policy interface is defined in `cpen455kv/policy_base.py`.
5. **Improve on H2O.** Implement `MyPolicy` and configure it in `student/config.yaml`. Justify your design and compare it against the baselines.
6. **Evaluate and explain.** Produce F1–F3, T1 and the failure case, and answer the report questions (see [Report and milestones](#report-and-milestones)).

Every policy must follow these rules:

- **Inputs.** A policy may use keys, values, positions and attention probes. It must not inspect text, token IDs, answers or hidden data.
- **Cost.** Probing, sorting and gathering all count toward the evaluated cost.
- **Eviction.** Return eviction indices only when the cache is over capacity. The indices must be distinct, and the cache must satisfy the budget after eviction.
- **Interface and state.** Keep the provided interfaces unchanged, and clear any per-sequence state between batches.

`cpen455kv/policy_base.py` includes full-cache and sliding-window policies as examples.

## Model evaluation

Every core submission uses the same base model and is evaluated at three cache budgets, **B = 64, 128 and 256** tokens. The harness streams each context through your policy and then scores the answer options without further eviction.

**Which questions count.** At budget $B$, a question is scored only if its context has at least $B + 2$ tokens; shorter contexts never force an eviction. This set depends only on $B$, so at each budget every policy is scored on exactly the same questions.

**Metrics.** For each budget $B$, the harness computes

$$
\mathrm{Acc}(B) = \frac{\text{correct answers}}{\text{scored questions}}
$$

$$
Q(B) = \min\left( 1, \frac{\mathrm{Acc}(B)}{\mathrm{Acc}_{\mathrm{full}}(B)} \right)
\qquad
R(B) = \min\left( 1, \frac{\mathrm{Throughput}_{\mathrm{policy}}(B)}{\mathrm{Throughput}_{\mathrm{window}}(B)} \right)
$$

and combines them into a local comparison score, averaged over the three budgets:

$$
\mathrm{Score} = \frac{1}{3} \sum_{B} Q(B) \cdot R(B)
$$

$Q$ measures retained accuracy relative to full cache on the same questions. $R$ measures throughput relative to the sliding-window baseline. Both are capped at 1. If full cache answers no scored questions correctly, the harness sets $Q = 1$; this case is uninformative and should be identified in your analysis.

The harness also reports diagnostics: a 95% item-bootstrap confidence interval for $Q$, and the changes in answer log-likelihood and context-token loss relative to the full cache.

**Timing.** Throughput is measured at batch size 64 and latency at batch size 1. Each measurement uses a 128-token prompt, 32 warm-up steps and 512 timed steps, and reports the median of three repeats. All policy work runs inside the timed region. The harness also reports model and policy GFLOPs and peak allocated GPU memory.

Follow the cache/interface rules, preserve full-cache predictions before eviction can affect them, and keep total GFLOPs per token within the full-cache cost. The harness reports diagnostics to help check these requirements. Local baseline rows use your implementations, so fix their bugs before interpreting comparisons. `--skip-timing` is useful for quick quality checks but does not measure efficiency.

**Leaderboard benchmark: TBA.** The benchmark, submission procedure and ranking details will be announced on Canvas. The local metrics above support development and report experiments; they do not assign official benchmark marks.

## Grading rubric

The project is worth **100 core points plus at most 10 bonus points**. The overall breakdown is below; the [grading rubric](guidance/grading.md) gives the component allocations and evidence assessed.

| Component | Points |
|---|---:|
| Conceptual understanding | 21 |
| Core implementation | 12 |
| Mutant hunt | 9 |
| Required experiments | 12 |
| Hidden benchmark | 20 |
| Efficiency and compliance | 8 |
| Analysis and failure case | 13 |
| Reproducibility | 5 |
| **Core total** | **100** |

## Bonus points

Choose **one** study, worth **up to +10 points**; multiple studies do not stack. We assess the hypothesis, implementation, controlled experiments, interpretation and reproducibility. Well-supported negative results are welcome. Keep bonus work separate from the unchanged core model and submission.

| Study | Required work |
|---|---|
| **MTP fine-tuning** | Add and train a multi-token prediction (MTP) component, and state whether the backbone is frozen or fine-tuned jointly. Compare ordinary greedy decoding, untrained MTP and trained MTP. Report held-out prediction quality, end-to-end decoding speed, peak memory and agreement with the same checkpoint's ordinary greedy decoder. If you use speculative verification, include the cost of verification and of rejected drafts. If the backbone changes, also report quality against the original checkpoint. |
| **Learned importance scorer** | Train a tiny MLP (at most 10,000 trainable parameters) on key statistics from the `cbt-val` fitting subset, with the backbone frozen. Compare it with H2O and with the same policy using an untrained scorer at all three budgets, and report Q, R, Score and scorer overhead. Every feature used at inference must be allowed by the cache-policy interface. |
| **Greedy-generation fidelity** | Measure how many generated tokens a compressed cache reproduces before its greedy continuation first differs from the full cache's. Compare Window, StreamingLLM, H2O and MyPolicy at all three budgets, and study the effect of changing one policy setting. Use the identical frozen checkpoint, prompt, dtype and deterministic argmax rule for every run. |

**MTP reference.** You may base your design on the [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2) (Section 2.2 and Figure 3). Adapt its MTP approach to GPT-2, cite the paper and explain what you changed.

**Fidelity protocol.** Number the generated tokens from 1. If the first mismatch with the full-cache continuation occurs at token $d$, the matching-prefix length is $m = d - 1$. If all **256** generated tokens match, record $m = 256$ and report "no divergence observed within the horizon", not a mismatch at token 257. Report the mean and median of $m$, a 95% bootstrap interval for the mean, and the fraction of continuations that match all 256 tokens. Compare independently generated greedy continuations, not teacher-forced next-token agreement.

Bonus points are allocated to hypothesis **1**, implementation and measurement **3**, controlled evaluation **3**, interpretation **2**, and reproducibility **1**. Follow the [bonus experiment protocol](guidance/guidance.md#bonus-experiments) and include an ablation, a limitation and a concrete next experiment.

## Submission requirements

Submit **one ZIP file** on Canvas. Its root must contain exactly the entries below and nothing else: no enclosing folder, no `__pycache__`, `__MACOSX` or hidden files.

| Path | Content |
|---|---|
| `student/` | `kv_cache.py`, `policy.py`, `config.yaml`, `part1_answers.yaml`, `MUTANTS.md`, `part2_results.json` and `my_tests/` (at least one `test_*.py`). Source and text files only (`.py`, `.yaml`, `.yml`, `.json`, `.md`, `.txt`), at most 1 MiB in total. |
| `validation.json` | The complete, timed `cbt-test` run of your final `MyPolicy` with the locked dependencies: `python -m cpen455kv.grade --strict-env --split cbt-test --out validation.json`. Rerun it after every change to your code or `config.yaml`. The harness records software, hardware, dtype and checkpoint metadata. |
| `report.pdf` | Your report (see [Report and milestones](#report-and-milestones)). |
| `AI_USE.md` | Your AI-use disclosure, filled in; write `none` on its own line if you used no AI tools. |
| `bonus/` | Optional: a `README.md` with exact commands, plus code and config, data-subset IDs, seeds, raw results and any trained state dictionaries. Keep bonus checkpoints out of `student/`, limit each to 1.6 GB, and use the large-file route announced on Canvas if needed. |

Do not include environments, base checkpoints, data, package caches, or copies of `cpen455kv/`, `nanogpt/` or `tests/`. For the core project, do not change package versions, model weights or the evaluation procedure.

**Check before you submit.** Build the ZIP from the project root (add `bonus` to the list if you did a bonus study), then check the exact file you will upload:

```bash
zip -r submission.zip student validation.json report.pdf AI_USE.md -x '*__pycache__*' '*.DS_Store'
python -m cpen455kv.check submission.zip
```

Upload the ZIP only if the check prints `check passed`. It checks the layout, allowed files, required evidence and validation settings. While you work, `python -m cpen455kv.check .` checks the submission files in your project root.

## Report and milestones

Write the report in the NeurIPS style ([style files](https://neurips.cc/Conferences/2023/PaperInformation/StyleFiles)) and submit it as `report.pdf` inside your ZIP. It may be **at most four pages**, excluding references and appendices. Focus on clear evidence and explanation.

We recommend three sections:

1. **Model.** Describe your policy formally: its scoring rule, eviction schedule, budget allocation, and any learned component with its loss function.
   - Include a figure that you created yourself showing the main computation graph: how tokens are scored and evicted during decoding.
   - Use equations rigorously and concisely. Add an algorithm box if the method is hard to follow from text alone.
   - Distinguish your policy from H2O and the other baselines, ideally with a comparative figure or table.
2. **Experiments.** Present the required results below and answer Report Q1–Q6, labelled separately from the concept check. Also include at least one of: an ablation of a specific design choice; the training method and any special techniques, if you trained anything; quantitative and qualitative analysis of your results.
3. **Conclusion.** Summarise your key findings and contributions, then discuss limitations and concrete directions for future work.

Put bonus work and the prompt-response summaries of any AI assistance in clearly labelled appendices. The [report checklist](report/README.md) includes the experiment protocols and Report Q1–Q6.

| Item | Required content |
|---|---|
| F1 | Acc and Q vs. budget on `cbt-val` and `cbt-test` for Window, StreamingLLM, H2O and MyPolicy, with Q intervals; accuracy vs. answer distance (each question's `distance`: context tokens after the end of the answer's last complete mention) |
| F2 | Throughput at batch 64, latency at batch 1, and a prediction from bytes moved and memory bandwidth for your hardware; state the device and dtype and show the variation across timing repeats |
| F3 | Model and policy GFLOPs per token vs. budget |
| T1 | At B = 256, vary one H2O setting at a time: eviction interval `interval` in {1, 8, 32}, scoring window `window` in {1, 8, 32} and attention sinks `sink` in {0, 1, 4}; include TOVA and StreamingLLM for comparison |
| Failure case | One question the full cache answers correctly but MyPolicy gets wrong, with the position of the evicted token, attention evidence, option log-likelihoods and a controlled intervention |

**Citations and disclosure.**

- Cite every external dataset, paper, blog post and code snippet that informs your work: in the report, and for code, also in the source file.
- If you used AI assistance for coding or writing (for example ChatGPT, Claude or Copilot), mention it in the main text, summarise the prompts and responses in an appendix, and fill in `AI_USE.md`.
- Credit anyone who gave you informal feedback, and describe any internal code base you reused.

Deadlines are announced on Canvas. A suggested order of work is: setup and concept check; cache and mutant hunt; H2O; MyPolicy and experiments; final validation, report and submission check. Each component earns its own points, and bonus work is optional.

## Academic integrity

Work individually. You may use any PyTorch function available in the course environment and consult external resources, but you must properly acknowledge every source and piece of prior work you use. You remain responsible for understanding everything you submit.

The following violations result in a grade of **zero** for the project and are handled under UBC policy:

1. Extensive reuse or adaptation of existing methods or code bases, including a completed cache implementation, without proper citation in both your report and your code.
2. Using tools such as ChatGPT, Claude or Copilot to generate code without acknowledging it, including the details of the prompt-response interactions.
3. Attempting to access, extract or reconstruct the hidden benchmark or its answers; writing a policy that reads text, token IDs or answers; detecting the grader; or evading the budget, FLOPs or timing measurements.
4. Submitting fabricated results, including to the leaderboard.
5. Any other academic dishonesty, such as sharing code, model checkpoints, results or inference outputs with other students.

## References and licences

- [nanoGPT](https://github.com/karpathy/nanoGPT) and [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- [H2O](https://arxiv.org/abs/2306.14048), [StreamingLLM](https://arxiv.org/abs/2309.17453) and [TOVA](https://arxiv.org/abs/2401.06104).
- [Children's Book Test](https://arxiv.org/abs/1511.02301); data attribution in [data/README.md](data/README.md).
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2), Section 2.2 and Figure 3 (optional MTP bonus reference).

Course code: MIT ([LICENSE](LICENSE)). nanoGPT: MIT ([nanogpt/LICENSE](nanogpt/LICENSE)). CBT data: GNU FDL 1.3.
