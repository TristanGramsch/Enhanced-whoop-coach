from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime

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


def _write_features(outputs_dir: str, features: Dict[str, Any]) -> Dict[str, Any]:
    out_path = Path(outputs_dir) / "journal_features.json"
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)
    return {"journal_features_path": str(out_path)}


def _maybe_extract_hrv_guess(text: str) -> Optional[float]:
    # Look for explicit patterns like: "my hrv (tomorrow) will be 65", "hrv 70 ms", "I think hrv is 62"
    patterns = [
        r"hrv[^\d]{0,10}(?:tomorrow|next\s*day)?[^\d]{0,10}(?:will\s*be|is\s*going\s*to\s*be|should\s*be|is)?[^\d]{0,5}(\d{2,3})(?:\s*ms)?",
        r"(?:i\s*think|i\s*believe)[^\d]{0,20}hrv[^\d]{0,10}(\d{2,3})(?:\s*ms)?",
        r"(?:tomorrow|next\s*day)[^\d]{0,10}hrv[^\d]{0,10}(\d{2,3})(?:\s*ms)?",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            try:
                val = float(m.group(1))
                if 20 <= val <= 200:
                    return val
            except Exception:
                pass
    # Fallback: any standalone number with ms within 5 chars
    m2 = re.search(r"(\d{2,3})\s*ms", text.lower())
    if m2:
        try:
            val = float(m2.group(1))
            if 20 <= val <= 200:
                return val
        except Exception:
            pass
    return None


def _append_user_guess(outputs_dir: str, value: float, source: str = "journal") -> None:
    log_path = Path(outputs_dir) / "user_guess.json"
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "date": datetime.utcnow().date().isoformat(),
        "value": value,
        "source": source,
    }
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            if isinstance(data, list):
                data.append(record)
            else:
                data = [data, record]
        except Exception:
            data = [record]
    else:
        data = [record]
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)


def process_journal_text(journal_text: str, outputs_dir: str) -> Dict[str, Any]:
    features = extract_features(journal_text)
    # Try to extract an HRV guess from the journal
    guess = _maybe_extract_hrv_guess(journal_text)
    if guess is not None:
        _append_user_guess(outputs_dir, guess)
        features["user_guess_value"] = guess
    return _write_features(outputs_dir, features)


def run_journal_processing(outputs_dir: str, journal_path: Optional[str] | None = None, journal_text: Optional[str] | None = None) -> Dict[str, Any]:
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    # Priority: explicit text > provided file > sample
    if journal_text is not None and journal_text.strip():
        features = extract_features(journal_text)
        guess = _maybe_extract_hrv_guess(journal_text)
        if guess is not None:
            _append_user_guess(outputs_dir, guess)
            features["user_guess_value"] = guess
        return _write_features(outputs_dir, features)

    if journal_path is not None:
        path = Path(journal_path)
    else:
        path = Path(__file__).resolve().parent / "sample_journal.txt"

    if not path.exists():
        sample = (
            "Felt pretty good today. Slept well and woke up rested. "
            "Did an easy workout at the gym. A bit of stress at work but manageable."
        )
        features = extract_features(sample)
        guess = _maybe_extract_hrv_guess(sample)
        if guess is not None:
            _append_user_guess(outputs_dir, guess)
            features["user_guess_value"] = guess
    else:
        text = path.read_text(encoding="utf-8")
        features = extract_features(text)
        guess = _maybe_extract_hrv_guess(text)
        if guess is not None:
            _append_user_guess(outputs_dir, guess)
            features["user_guess_value"] = guess

    return _write_features(outputs_dir, features)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = run_journal_processing(outputs_dir=str(repo_root / "shared" / "outputs"))
    print(json.dumps(artifacts, indent=2))