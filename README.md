# Don't Believe Everything You Retrieve: RAG Poisoning Experiment

The current experiment uses [25 reviewed questions](data/final25/README.md)
selected from the [100-question source pool](data/sampled100/clean.json). Each
question has four supplied documents and conditions at 0%, 25%, 50%, and 75%
poisoning. See [POISONING.md](POISONING.md) for the editing policy, script roles,
and commands.

The three scripts for the final report are `prepare_dataset.py` (source-pool
preparation), `build_conditions.py` (replay reviewed edits and export conditions),
and `run_conditions.py` (Gemini answering). The canonical reviewed edits are
stored in `data/final25/approved_edits.json`.

## What the original baseline does
1. Loads the `distractor` validation split of HotpotQA.
2. Creates one candidate document per context article.
3. Ranks the **provided per-question candidate articles** with TF-IDF/cosine similarity.
4. Builds a **baseline-only** prompt (no verification or abstention defenses).
5. Calls Gemini to answer the questions.
6. Saves questions, ground truth, generated answers, and ranked documents as JSON.


## Set up
1. Install dependencies: python3 -m pip install datasets google-genai scikit-learn
2. Set up a Gemini API key
- Get a key from Google AI Studio.
- On Mac/Linux, run:
    read -s GEMINI_API_KEY
    export GEMINI_API_KEY
(Paste your API key when prompted. Do not upload API keys to GitHub.)
3. Run the baseline
- python3 rag.py
- The program currently tests 5 questions and retrieves 4 documents per question.
- Results are saved in outputs/baseline.json.


## Shared experiment
- Set `POISON_PERCENTAGE` in `run_conditions.py` to 0, 25, 50, or 75 and run `.\.venv\Scripts\python.exe run_conditions.py` after setting the API key. Each run produces 25 results and saves a separate results file.
- Validate or rebuild the reviewed conditions with `.\.venv\Scripts\python.exe build_conditions.py`.
- Implement the baseline, cross-source verification, conflict-aware abstention, and combined prompting strategies.
- Run the same questions across all experimental conditions.
- Calculate answer accuracy, attack success rate, and abstention rate.
- Compare performance on clean and poisoned documents.
- Keep the model, question IDs, and retrieval settings consistent across experiments so the results can be compared fairly.
