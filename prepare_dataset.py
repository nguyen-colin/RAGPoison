from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import platform

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NUM_QUESTIONS = 100
TOP_K = 4
RANDOM_SEED = 42
OUTPUT_DIR = Path("data/sampled100")


def select_indices(size):
    """Try the original random sample first, then a seeded remaining order."""
    if not 1 <= NUM_QUESTIONS <= size:
        raise ValueError("Question count must be between 1 and the dataset size")
    initial = random.Random(RANDOM_SEED).sample(range(size), NUM_QUESTIONS)
    initial_set = set(initial)
    remaining = [i for i in range(size) if i not in initial_set]
    random.Random(RANDOM_SEED).shuffle(remaining)
    return initial + remaining


def retrieve(example, top_k):
    titles = example["context"]["title"]
    paragraphs = example["context"]["sentences"]
    if len(titles) != len(paragraphs) or len(titles) < top_k:
        raise ValueError(f"Invalid candidate count for {example['id']}")
    documents = [title + " " + " ".join(sentences)
                 for title, sentences in zip(titles, paragraphs)]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)
    scores = cosine_similarity(vectorizer.transform([example["question"]]), vectors)[0]
    best = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
    facts = example["supporting_facts"]
    evidence = []
    for title, sentence_id in zip(facts["title"], facts["sent_id"]):
        index = titles.index(title)
        evidence.append({"title": title, "sentence_id": sentence_id,
                         "text": paragraphs[index][sentence_id], "retrieved": index in best})
    return {
        "question_id": example["id"], "question": example["question"],
        "correct_answer": example["answer"], "top_k": top_k,
        "type": example.get("type"), "level": example.get("level"),
        "retrieved_documents": [
            {"title": titles[i], "text": documents[i], "score": round(float(scores[i]), 3),
             "candidate_index": i, "sentences": paragraphs[i]}
            for i in best
        ],
        "evaluation_only": {
            "supporting_facts": evidence,
            "all_supporting_articles_retrieved": all(fact["retrieved"] for fact in evidence),
        },
    }


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Dataset already exists: {OUTPUT_DIR}. Change OUTPUT_DIR to reproduce separately.")
    if NUM_QUESTIONS < 1 or TOP_K < 1:
        raise ValueError("Question count and TOP_K must be positive")
    from datasets import load_dataset

    print("Loading HotpotQA distractor validation; no model calls.", flush=True)
    raw = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    fingerprint = getattr(raw, "_fingerprint", None)
    source = {"dataset": "hotpotqa/hotpot_qa", "configuration": "distractor", "split": "validation"}
    indices = select_indices(len(raw))
    records = []
    ineligible = []
    screening = []
    scanned_indices = []
    for source_index in indices:
        example = raw[source_index]
        scanned_indices.append(source_index)
        if len(example["context"]["title"]) < TOP_K:
            ineligible.append({"question_id": example["id"],
                               "reason": "fewer_candidate_articles_than_top_k", "example": example})
            screening.append({"question_id": example["id"], "source_index": source_index,
                              "decision": "insufficient_candidates"})
        else:
            record = retrieve(example, TOP_K)
            if not record["evaluation_only"]["all_supporting_articles_retrieved"]:
                screening.append({"question_id": example["id"], "source_index": source_index,
                                  "decision": "missing_supporting_articles"})
            else:
                records.append(record)
                screening.append({"question_id": example["id"], "source_index": source_index,
                                  "decision": "selected"})
        if len(records) == NUM_QUESTIONS:
            break
    if len(records) != NUM_QUESTIONS:
        raise ValueError("Not enough questions meet the supporting-article requirement")
    ids = [r["question_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    content = (json.dumps(records, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    complete = sum(r["evaluation_only"]["all_supporting_articles_retrieved"] for r in records)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "dataset_fingerprint": fingerprint,
        "selection": "sample without replacement",
        "sampling": "random", "seed": RANDOM_SEED,
        "source_population_size": len(raw), "selected_source_indices": [s["source_index"] for s in screening if s["decision"] == "selected"],
        "sampled_question_ids": [s["question_id"] for s in screening],
        "scanned_source_indices": scanned_indices,
        "require_supporting_evidence": True,
        "screening_file": "screening.json",
        "scanned_questions": len(screening), "ineligible_questions": len(ineligible),
        "ineligible_file": "ineligible.json",
        "num_questions": len(records), "question_ids": ids, "top_k": TOP_K,
        "retrieval": "per-question TF-IDF, English stop words, cosine similarity; ties by candidate index",
        "clean_file": "clean.json", "clean_sha256": hashlib.sha256(content).hexdigest(),
        "all_supporting_articles_retrieved_count": complete,
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("datasets", "scikit-learn")},
        "notes": "Coverage is not a manual sufficiency judgment. See screening.json for explicit coverage filtering. No selection based on generator output or answering-model success. No poisoned conditions generated.",
    }
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "clean.json").write_bytes(content)
    (OUTPUT_DIR / "ineligible.json").write_text(json.dumps(ineligible, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "screening.json").write_text(json.dumps(screening, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} questions to {OUTPUT_DIR}")
    print(f"All supporting articles retrieved: {complete}/{len(records)}.")
    print(f"Screened {len(screening)} questions; decisions saved in screening.json.")


if __name__ == "__main__":
    main()