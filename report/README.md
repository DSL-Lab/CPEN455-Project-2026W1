# Report checklist

Submit `report.pdf` inside your submission ZIP: NeurIPS style ([style files](https://neurips.cc/Conferences/2023/PaperInformation/StyleFiles)), at most four pages excluding references and appendices. We recommend three sections: Model, Experiments and Conclusion. See the [README](../README.md#report-and-milestones) for the guidelines.

**Model**

- [ ] Formal description of your policy: scoring rule, eviction schedule, budget allocation, and any learned component with its loss.
- [ ] A computation-graph figure you created yourself.
- [ ] Concise equations; an algorithm box if the method is hard to follow from text.
- [ ] Comparison with H2O and the other baselines (figure or table).

**Experiments**

- [ ] Part 2 table: vanilla, full cache and 256-token window; batch 1 and 16; 512 generated tokens; memory, tokens/s and GFLOPs.
- [ ] F1: Acc/Q vs budget on `cbt-val` and `cbt-test`, Q intervals and accuracy vs answer distance.
- [ ] F2: throughput/latency, timing variation and bandwidth prediction.
- [ ] F3: model/policy GFLOPs separately.
- [ ] T1: controlled H2O sweeps at B = 256, plus TOVA/StreamingLLM comparisons.
- [ ] Failure case: a question full cache gets right and MyPolicy gets wrong; show the relevant eviction, attention, option scores and a controlled intervention.
- [ ] Report Q1–Q6, labelled separately from the concept check.
- [ ] At least one of: an ablation of a design choice; training method and special techniques (if you trained anything); quantitative and qualitative analysis.

**Conclusion**

- [ ] Key findings and contributions.
- [ ] Limitations, a concrete follow-up experiment, and reproducibility details (GPU/dtype/checkpoint/config/data IDs).

**Citations, disclosure and appendices**

- [ ] Every external dataset, paper, blog post and code snippet cited (code also cited in the source).
- [ ] AI assistance mentioned in the main text, prompt-response summaries in an appendix, and `AI_USE.md` filled in.
- [ ] Informal feedback credited; reused internal code bases described.
- [ ] Optional bonus appendix: one study, its hypothesis, protocol, results and limitations; separate `bonus/` artifacts.

Keep mutant explanations and regression tests in `student/MUTANTS.md` and `student/my_tests/`. Before uploading, run `python -m cpen455kv.check submission.zip`.

## Experiment protocols

Use the same item IDs for compared methods, label axes/units, and include raw results. Record model, device, dtype, configuration and item IDs. Core experiments use unchanged GPT-2-medium.

- **Part 2:** compare vanilla nanoGPT, full cache and a 256-token window at batch sizes 1 and 16, generating 512 tokens. Report tokens/s, peak memory and GFLOPs per token for every row.
- **F1–F3:** compare Window, StreamingLLM, H2O and MyPolicy at budgets 64, 128 and 256. For F1 use both complete public splits, include 95% item-bootstrap Q intervals, and state distance-bin boundaries and counts. For F2 use the harness timing defaults (batch 64 throughput, batch 1 latency), report the median and range of three repeats, and estimate runtime from bytes moved and your device's bandwidth. Separate model and policy GFLOPs in F3.
- **T1:** use B = 256 and baseline H2O `sink=4, recent=32, window=8, interval=8`. Change one factor at a time: interval {1,8,32}, window {1,8,32}, sinks {0,1,4}. For TOVA use `sink=0, recent=1, window=1, interval=1`; compare StreamingLLM with `sink=4`. Use the first 128 items from each public split in file order, or larger fixed subsets identified in the report. Report Q(256) with intervals, R(256), and their product Q(256) × R(256) using standard timing settings; this is the contribution at one budget, while Score averages all three budgets. Keep all tuning decisions on `cbt-val`.
- **Failure case:** trace why MyPolicy loses a correct full-cache answer. Test the explanation with a controlled intervention, such as retaining the relevant token.

## Report Q1–Q6

Answer from your measurements, separately from the concept check. No particular performance ordering is assumed.

1. Explain batch-1 latency and batch-64 throughput using bytes moved, bandwidth and observed runtime.
2. Compare FLOP reduction with runtime reduction. Account for attention, other model computation and policy overhead.
3. Can a low-attention token become useful later? Give a concrete context/query example.
4. What effect do attention sinks have in your runs? Use your sink ablation as evidence.
5. Does H2O improve Score over StreamingLLM? Explain the quality and probe-cost tradeoff using Q and R.
6. Where does MyPolicy differ from H2O, and why? Support the explanation with a controlled comparison, uncertainty and a concrete success or failure.
