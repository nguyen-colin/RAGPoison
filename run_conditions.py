from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


POISON_PERCENTAGE = 75  # Run 0, then 25, 50, and 75 in separate runs.
CONDITIONS_FILE = Path(f"data/final25/conditions_{POISON_PERCENTAGE}.json")
OUTPUT_FILE = Path(f"outputs/poisoning/final25-results-{POISON_PERCENTAGE}.json")
MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
REQUEST_DELAY_SECONDS = 10  # Pace calls; check your project's limits in AI Studio.


def load_conditions(path):
    conditions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Conditions must be a nonempty JSON list")
    seen = set()
    for condition in conditions:
        key = (condition["question_id"], condition["poison_percentage"])
        if key in seen:
            raise ValueError(f"Duplicate condition: {key}")
        seen.add(key)
        prompt = condition["model_input"]["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt: {key}")
        if not isinstance(condition["evaluation_only"], dict):
            raise ValueError(f"Missing evaluation metadata: {key}")
    return conditions


def run_condition(client, condition, model, temperature, api_error, sleep=time.sleep):
    """Only the prepared prompt is sent to Gemini; metadata stays local."""
    result = {
        **condition,
        "model": model,
        "temperature": temperature,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "generated_answer": None,
        "status": "error",
    }
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=condition["model_input"]["prompt"],
                config={"temperature": temperature},
            )
            result["generated_answer"] = response.text
            result["status"] = "ok" if response.text and response.text.strip() else "empty_response"
            break
        except api_error as error:
            # Keep errors as errors, rather than counting them as abstentions.
            result["error_code"] = error.code
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                break
            sleep(10 * (2 ** attempt))
    if result["status"] != "error":
        result.pop("error_code", None)
    return result


def main():
    if POISON_PERCENTAGE not in (0, 25, 50, 75):
        raise ValueError("POISON_PERCENTAGE must be 0, 25, 50, or 75")
    conditions = load_conditions(CONDITIONS_FILE)
    if len(conditions) != 25 or any(c["poison_percentage"] != POISON_PERCENTAGE for c in conditions):
        raise ValueError("Expected 25 questions at the selected poisoning percentage")
    if not MODEL.strip() or not 0 <= TEMPERATURE <= 2:
        raise ValueError("Set a nonempty MODEL and TEMPERATURE between 0 and 2")
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT_FILE}. Change OUTPUT_FILE to save another run.")
    print(f"Poisoning: {POISON_PERCENTAGE}%; model: {MODEL}; temperature: {TEMPERATURE}; questions: {len(conditions)}")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise ValueError("Set GEMINI_API_KEY in this terminal before running")
    from google import genai
    from google.genai.errors import APIError

    failures = 0
    completed = 0
    results = []
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with genai.Client() as client, OUTPUT_FILE.open("x", encoding="utf-8") as stream:
        stream.write("[]\n")
        stream.flush()
        for condition in conditions:
            if completed:
                time.sleep(REQUEST_DELAY_SECONDS)
            result = run_condition(client, condition, MODEL, TEMPERATURE, APIError)
            results.append(result)
            stream.seek(0)
            json.dump(results, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.truncate()
            stream.flush()
            completed += 1
            failures += result["status"] != "ok"
            print(f"{condition['poison_percentage']}% [{result['status']}]: {result['generated_answer']}")
            if result.get("error_code") == 429:
                print("Stopped after repeated rate-limit/quota errors. Check AI Studio before retrying; completed results are saved.")
                break
    print(f"Saved {completed}/{len(conditions)} results to {OUTPUT_FILE}; unsuccessful conditions: {failures}")
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
