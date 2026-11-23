# Asha's HotpotQA baseline RAG

This is a controlled RAG baseline for the project "Don't Believe Everything You Retrieve."

## What it does
1. Loads the `distractor` validation split of HotpotQA, or a local JSON file in equivalent schema.
2. Creates one candidate document per context article.
3. Ranks the **provided per-question candidate articles** with TF-IDF/cosine similarity.
4. Builds a **baseline-only** prompt (no verification or abstention defenses).
5. Optionally calls an OpenAI model through the Responses API.
6. Saves each run's question, ground truth, answer, ranked documents and prompt as JSONL.

**Important scope limitation:** This is *not* open-domain Wikipedia retrieval. It re-ranks HotpotQA's given distractor paragraphs. This is suitable for a controlled first experiment, but it makes a different claim than retrieval over a full external corpus.

## Set up
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

## Run without network/API key (local synthetic sample)
```bash
python rag.py --input-json data/sample_hotpot.json --dry-run --output outputs/sample.jsonl
```

## Load 5 real HotpotQA questions (requires Internet, not an API key)
```bash
python rag.py --count 5 --dry-run --output outputs/hotpot_dry_run.jsonl
```

## Generate answers with the OpenAI API
Set `OPENAI_API_KEY` in your shell or use a local ignored `.env` with a loader of your choice (the code does NOT automatically load `.env`). Example:
```bash
export OPENAI_API_KEY="your-key-here"       # macOS/Linux
# $env:OPENAI_API_KEY="your-key-here"      # PowerShell
python rag.py --count 5 --model gpt-4.1-mini --output outputs/baseline.jsonl
```
Model availability and billing depend on your API account. Keep the exact model fixed across experiments.

## Compatible with teammates' modified documents
Colin can provide a HotpotQA-format JSON containing the same question IDs and `context` structure but altered document sentences. Use:
```bash
python rag.py --input-json data/poisoned_hotpot.json --top-k 4 --output outputs/poisoned_baseline.jsonl
```
No changes to the generator or retriever are required. Dan and Isaac can replace `build_baseline_prompt` while retaining `retrieve` and the same inputs. Jake can call `run_one` in a loop or invoke the CLI. Ivan can read `ground_truth`, `generated_answer`, `question_id`, and document provenance from JSONL.

## Experimental caution
- Use exactly the same questions, order, model, `top_k`, retrieval code, and API settings for all conditions.
- Document poisoning percentages **in candidate documents** do not guarantee the same percentages among top-k **retrieved** documents. Log the actual poisoned share among retrieved docs.
- At 0%, 25%, 50%, 75%, a fixed four-document retrieved context can attain these exact fractions, but the candidate pool may have a different size. Coordinate condition definitions with Colin.
- This baseline intentionally has no cross-source verification or abstention instruction. Its answers might still abstain naturally; that is an observed behavior rather than a programmed defense.
- Avoid leaking the ground-truth answer to the LLM; it is saved for evaluation only.
- Save retrieved context and prompt to allow debugging and reproducibility.
- Output JSONL may contain full context and should be treated as an experiment artifact.
