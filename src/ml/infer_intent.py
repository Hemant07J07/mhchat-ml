from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from repo root:
#   python src\ml\infer_intent.py
# (When executed this way, Python sets sys.path[0] to src/ml, so `import src...`
# would fail unless we add the project root.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.utils_intent import load_intent_pipeline
from src.ml.safety_checks import contains_crisis_words

nlp, id2label = load_intent_pipeline("src/models/intent")
print("Model pipeline loaded. id2label:", id2label)

# Replace / add examples to test
examples = [
    "I feel so hopeless and tired of living.",
    "Can you tell me a breathing excersie?",
    "What's up?",
    "I have been thinking about killing myself",
    "I feel anxious about the exam tomorrow"
]

for ex in examples:
    raw = nlp(ex)

    # Normalize output across transformers versions.
    # Possible shapes:
    # - dict (single prediction)
    # - list[dict] (all labels)
    # - list[list[dict]] (batch of inputs)
    if isinstance(raw, dict):
        scores = [raw]
    elif isinstance(raw, list) and raw and isinstance(raw[0], list):
        scores = raw[0]
    else:
        scores = raw

    if not isinstance(scores, list) or not scores or not isinstance(scores[0], dict):
        raise TypeError(f"Unexpected pipeline output: {type(raw)} -> {raw!r}")

    # find best
    best = max(scores, key=lambda d: float(d.get("score", 0.0)))
    label = best["label"]
    score = best["score"]
    # HF pipeline returns label names like "LABEL_0" when return_all_scores True.
    # If id2label exists, try to map.
    human_label = None
    try : 
        # label may be "LABEL_2" -> extract index
        if isinstance(label, str) and label.upper().startswith("LABEL_"):
            idx = int(label.split("_")[-1])
            human_label = id2label.get(str(idx), None) if id2label else None
    except Exception:
        human_label = None

    if human_label is None:
        human_label = label # fallback to whatever label is given

    if contains_crisis_words(ex):
        print("SAFETY: Crisis keyword detected -> force crisis flag")
        human_label = "crisis"
        score = 0.99
    
    print(f"INOUT: {ex}")
    print(f"PREDICTED: {human_label} ({score:.3F})")
    print("-" * 50)