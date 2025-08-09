from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, Any

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "happy", "relaxed", "rested", "energized", "productive",
}
NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "stress", "stressed", "tired", "exhausted", "sick", "injured",
}

SLEEP_KEYWORDS = {"sleep", "slept", "bed", "rest"}
TRAINING_KEYWORDS = {"workout", "train", "run", "bike", "swim", "lift", "gym"}


def tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_features(text: str) -> Dict[str, Any]:
    tokens = tokenize(text)
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    sentiment = (pos - neg) / max(len(tokens), 1)

    sleep_mentions = any(t in SLEEP_KEYWORDS for t in tokens)
    training_mentions = any(t in TRAINING_KEYWORDS for t in tokens)

    features = {
        "token_count": len(tokens),
        "positive_count": pos,
        "negative_count": neg,
        "sentiment_score": sentiment,
        "mentions_sleep": sleep_mentions,
        "mentions_training": training_mentions,
        "raw_excerpt": text[:500],
    }
    return features


def run_journal_processing(outputs_dir: str, journal_path: str | None = None) -> Dict[str, Any]:
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    # Prefer provided journal file; else fallback to sample
    if journal_path is not None:
        path = Path(journal_path)
    else:
        path = Path(__file__).resolve().parent / "sample_journal.txt"

    if not path.exists():
        # If nothing provided, create a minimal sample
        sample = (
            "Felt pretty good today. Slept well and woke up rested. "
            "Did an easy workout at the gym. A bit of stress at work but manageable."
        )
        features = extract_features(sample)
    else:
        text = path.read_text(encoding="utf-8")
        features = extract_features(text)

    out_path = outputs / "journal_features.json"
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)

    return {"journal_features_path": str(out_path)}


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = run_journal_processing(outputs_dir=str(repo_root / "shared" / "outputs"))
    print(json.dumps(artifacts, indent=2))