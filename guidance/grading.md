# Grading rubric

The project has **100 core points**, plus **up to 10 bonus points** for one optional study. This rubric gives the allocations and the evidence assessed. The [README](../README.md) defines the tasks; the [report checklist](../report/README.md#experiment-protocols) gives the experiment protocols.

| Component | Points | Allocation and evidence |
|---|---:|---|
| Conceptual understanding | 21 | Concept check **6** (Q1–Q4: **1, 2, 1, 2**); Report Q1–Q6 **15** (**3, 2, 2, 2, 3, 3**). Explain the mechanisms and support experimental claims with measurements. |
| Core implementation | 12 | Cache/generation and Part 2 measurements **4**; H2O **6**; explanation of selected MyPolicy code **2**. Assess correctness, interface contracts and understanding. |
| Mutant hunt | 9 | **1 per bug**: working fix **0.5**, documented cause and regression test **0.5**. TODO implementations are separate from the nine planted bugs. |
| Required experiments | 12 | F1 **4**; F2 **3**; F3 **2**; T1 **3**. Assess complete, controlled comparisons with the required metrics and uncertainty. |
| Hidden benchmark | 20 | Policy performance on held-out data; benchmark details TBA. |
| Efficiency and compliance | 8 | Cache/interface correctness **3**; FLOPs budget **3**; runtime overhead **2**. The criteria below apply. |
| Analysis and failure case | 13 | MyPolicy failure case **5**; interpretation of F1/F2 and Part 2 **5**; supported limitations and a concrete follow-up experiment **3**. |
| Reproducibility | 5 | Fresh-environment execution **1**; submission ZIP check **1**; complete, reproducible validation **3**, one per budget. |
| **Core total** | **100** | |

For written work and experiments, full credit requires correct, complete evidence; materially correct but incomplete work earns partial credit within the stated allocation. Implementation marks reflect the fraction of correctness checks passed. Missing or incorrect work earns no credit. Points do not transfer between components; round only the final total to two decimal places.

## Efficiency and compliance (8)

- **Cache rules (3):** legal, distinct retained indices and the required budget **1**; eviction only when over capacity **1**; pre-eviction prediction agreement **1**. Each criterion must hold at all budgets.
- **FLOPs (3):** **1 per budget** for keeping total model and policy computation within the full-cache cost.
- **Overhead (2):** efficient policy execution, supported by measured runtime against Window.

Missing or failed measurements earn no points for the affected criterion. A quality-only run cannot earn FLOPs or overhead marks.

## Analysis and failure case (13)

The failure case must be a question that **full cache answers correctly and MyPolicy answers incorrectly**. Identify the relevant eviction, show attention evidence and both policies' option log-likelihoods, and test the explanation with a controlled intervention. Interpret the experimental trends and uncertainty, and connect limitations to evidence and a concrete next experiment. Detailed experiment instructions stay in the [report checklist](../report/README.md).

## Reproducibility (5)

Fresh-environment execution earns **1**, a passing submission ZIP check earns **1**, and complete, reproducible validation earns **3**, one per budget. Supply the source, configuration and results needed to repeat your experiments in the locked course environment. Record the checkpoint, device, dtype and software metadata, and explain any observed variation between runs.

## Bonus (10 maximum)

Choose one of the [bonus studies](../README.md#bonus-points): hypothesis **1**; implementation and measurement **3**; controlled evaluation **3**; interpretation **2**; reproducibility **1**. Follow the [common protocol](guidance.md#bonus-experiments), include an ablation and limitation, and submit the code, commands and raw results. A well-supported negative result can earn full credit. Bonus studies do not stack or alter core benchmark marks.
