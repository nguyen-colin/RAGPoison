# Final25 files

This folder contains 25 questions selected from `data/sampled100/`, 75 approved poisoned document versions, and 100 condition inputs. It extends the previous 20-question cohort with five questions about film direction, a creator's birthplace, an NBA draft venue, magazine chronology, and documentary chronology. The previous 20 records and their prompts are unchanged. See [POISONING.md](../../POISONING.md) for the policy and workflow.

| File | Purpose |
|---|---|
| `clean.json` | The 25 questions, correct answers, and four original sources per question. |
| `approved_edits.json` | Original and poisoned text, exact replacement spans, target answers, types, reasons, approvals, and poisoning order. |
| `conditions.json` | All 100 prepared inputs: the same 25 questions at four poisoning levels. |
| `conditions_0.json` | Twenty-five inputs with four unchanged sources each. |
| `conditions_25.json` | Twenty-five inputs with one poisoned and three unchanged sources each. |
| `conditions_50.json` | Twenty-five inputs with two poisoned and two unchanged sources each. |
| `conditions_75.json` | Twenty-five inputs with three poisoned and one unchanged source each. |
| `review.json` | Evidence reasoning, review decisions, and limitations. |
| `manifest.json` | Cohort IDs, the five added IDs, provenance, counts, edit types, and file checksums. |
