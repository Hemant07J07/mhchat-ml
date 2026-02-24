import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve


# Allow running as:
#   python -m src.ml.compute_thresholds
# or directly:
#   python src/ml/compute_thresholds.py
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.ml.fast_infer import predict


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute per-class score thresholds from an eval split")
    parser.add_argument(
        "--data-file",
        default="data/intent_data.csv",
        help="CSV with columns: text,label",
    )
    parser.add_argument(
        "--model-path",
        default="src/models/intent",
        help="Folder containing label2id.json (and the intent model)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data used as eval split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.9,
        help="Pick threshold that achieves at least this recall (per class) with best precision",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to write thresholds JSON (default: <model-path>/threshold.json)",
    )
    args = parser.parse_args()

    output_file = args.output_file or os.path.join(args.model_path, "threshold.json")

    with open(os.path.join(args.model_path, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {str(k): int(v) for k, v in label2id.items()}
    num_labels = max(label2id.values()) + 1 if label2id else 0
    if num_labels <= 0:
        raise RuntimeError(f"No labels found in {args.model_path}/label2id.json")

    from datasets import load_dataset

    dataset = load_dataset("csv", data_files=args.data_file, split="train")
    dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    eval_ds = dataset["test"]

    all_probs: list[list[float]] = []
    y_true: list[int] = []

    for text, label in zip(eval_ds["text"], eval_ds["label"]):
        label_id = label2id.get(str(label))
        if label_id is None:
            continue
        res = predict(text)
        probs = res.get("all_probs")
        if not isinstance(probs, list) or len(probs) != num_labels:
            continue
        all_probs.append([float(x) for x in probs])
        y_true.append(int(label_id))

    if not all_probs:
        raise RuntimeError("No eval examples produced probabilities; check your data and model")

    all_probs_arr = np.asarray(all_probs, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    thresholds: dict[str, float] = {}
    for class_id in range(num_labels):
        y_bin = (y_true_arr == class_id).astype(int)
        probs = all_probs_arr[:, class_id]
        precision, recall, thresh = precision_recall_curve(y_bin, probs)

        idx = np.where(recall >= float(args.target_recall))[0]
        if len(idx) > 0:
            pick = idx[np.argmax(precision[idx])]
            chosen = float(thresh[pick]) if pick < len(thresh) else 0.5
        else:
            chosen = 0.5

        thresholds[str(class_id)] = float(chosen)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, sort_keys=True)

    print(f"Wrote {len(thresholds)} thresholds to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())