# Poisoning policy and workflow

## Dataset selection

The source pool in `data/sampled100/` contains 100 HotpotQA distractor validation questions selected using seed 42 and evidence-coverage screening. We screened 161 questions to obtain 100 whose top four TF-IDF documents include all annotated supporting articles. From this pool, we selected 25 questions for AI-assisted curation based on evidence and editing feasibility, without using observed attack success. The final set in `data/final25/` is a curated sample, not a random sample of 25.

## Allowed edits

Choose one intended wrong answer per question. Prefer changing a relevant fact, such as a date, number, name, or location; when needed, rewrite a short passage or add one false, answer-relevant sentence, including in a distractor. Preserve titles, IDs, order, scores, original text, and surrounding content. Use one contiguous replacement per document and add no more than 60 words. Record the replacement spans, edit type, and reason. All three poisoned documents must support the same wrong answer without canceling one another. Do not add model instructions or fabricate claims of independent verification.

## Review and records

Check the clean evidence, target meaning, grammar, relevance, and consistency before approving each edited document. Record decisions and limitations in `review_notes` or `review.json`, and keep replacement spans consistent with the final text. Approved edits and their order are stored in `data/final25/approved_edits.json`. Approval means an edit is suitable for the experiment; it does not mean the attack succeeded. Keep unsuccessful attacks in the results and keep answers, targets, and edit labels outside the answering prompt.

## Poisoning levels

Each question supplies four documents: 0% uses four originals; 25% replaces the first approved document; 50% replaces the first two; and 75% replaces all three. Follow `poison_order`, keeping the same questions and document order across levels and prompting strategies. The percentage counts documents, not changed words. An unchanged source may be a distractor. The final dataset contains 25 questions, 75 poisoned alternatives, and 100 condition inputs.

## prepare_dataset.py

This script loads HotpotQA, samples questions, ranks articles using TF-IDF/cosine similarity, and saves the 100-question pool with its selection audit. It makes no Gemini calls. Settings are fixed at the top: 100 questions, seed 42, and four documents per question. Run `.\.venv\Scripts\python.exe prepare_dataset.py` only to reproduce preparation; because `data/sampled100/` already exists, change `OUTPUT_DIR` first to save a separate copy.

## build_conditions.py

This script reads `data/final25/approved_edits.json`, validates the original sources and recorded replacements against `data/sampled100/clean.json`, and builds all 100 prompts. It uses existing approved edits rather than generating new false claims, and makes no Gemini calls. Run `.\.venv\Scripts\python.exe build_conditions.py` to save a rebuilt copy to `outputs/poisoning/final25-rebuilt.json`. Existing output files are protected; paths and poisoning levels are settings at the top of the script.

## run_conditions.py

Set `POISON_PERCENTAGE` at the top of `run_conditions.py` to 0, 25, 50, or 75, then run `.\.venv\Scripts\python.exe run_conditions.py` in the terminal where `GEMINI_API_KEY` is set. Each run reads `data/final25/conditions_PERCENTAGE.json` and saves `outputs/poisoning/final25-results-PERCENTAGE.json`, producing 25 results per percentage. The model remains `gemini-3.5-flash-lite` at temperature 0, with a ten-second pause between questions. Persistent 429 errors stop the run after saving completed results; existing outputs are preserved and automatic resume is not implemented. Each new run answers all 25 questions at that level; earlier-result loading has been removed. Only `model_input.prompt` is sent, keeping evaluation metadata local. API errors and empty responses are separate from abstentions. Run all four levels using the same questions, model, and settings.
