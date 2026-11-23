# Don't Believe Everything You Retrieve : RAG Poisoning Experiemnt  

## What our RAG pipeline does
1. Loads the `distractor` validation split of HotpotQA, or a local JSON file in equivalent schema.
2. Creates one candidate document per context article.
3. Ranks the **provided per-question candidate articles** with TF-IDF/cosine similarity.
4. Builds a **baseline-only** prompt (no verification or abstention defenses).
5. Optionally calls an OpenAI model through the Responses API.
6. Saves each run's question, ground truth, answer, ranked documents and prompt as JSONL.


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


## Experimental caution
- Add poisoned document conditions (0%, 25%, 50%, and 75%).
- Implement the baseline, cross-source verification, conflict-aware abstention, and combined prompting strategies.
- Run the same questions across all experimental conditions.
- Calculate answer accuracy, attack success rate, and abstention rate.
- Compare performance on clean and poisoned documents.
- Keep the model, question IDs, and retrieval settings consistent across experiments so the results can be compared fairly.
