import os
import json
import argparse
import sys
from pathlib import Path

# Ensure the mhchat-ml repo root is importable (so `import src...` works even if
# this script is launched from a different working directory).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: 'datasets' (HuggingFace).\n"
        f"Python interpreter in use: {sys.executable}\n\n"
        "Fix (use the mhchat-ml virtualenv):\n"
        f"  cd {str(_REPO_ROOT)}\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m pip install -r requirements.txt\n"
    ) from exc
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
import sklearn.metrics as skm

from sklearn.utils.class_weight import compute_class_weight
from src.ml.weighted_trainer import WeightedTrainer

# Config - tuned for CPU / low RAM
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 8       # small to fit RAM
NUM_EPOCHS = 2       # keep low for quick runs
OUTPUT_DIR = "src/models/intent"

parser = argparse.ArgumentParser(description="Train intent classifier (class-weighted)")
parser.add_argument(
    "--data-file",
    default="data/intent_data.csv",
    help="Path to CSV with columns: text,label",
)
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load CSV dataset
print(f"Loading dataset from {args.data_file}")
dataset = load_dataset("csv", data_files=args.data_file, split="train")

# Small validation split for metrics tracking
dataset = dataset.train_test_split(test_size=0.15, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# Create label mapping automatically (from both splits)
labels = sorted(list(set(train_ds["label"]) | set(eval_ds["label"])))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print("labels found:", labels)
print("label2id:", label2id)

# Map labels to integers for training
def map_labels(example):
    example["labels"] = label2id[example["label"]]
    return example

train_ds = train_ds.map(map_labels)
eval_ds = eval_ds.map(map_labels)

# Tokenize
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=MAX_LENGTH)

train_ds = train_ds.map(preprocess, batched=True)
eval_ds = eval_ds.map(preprocess, batched=True)

# Set format for PyTorch
train_ds = train_ds.remove_columns(["text", "label"])
eval_ds = eval_ds.remove_columns(["text", "label"])
train_ds.set_format(type="torch")
eval_ds.set_format(type="torch")

# Data collator to handle dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model (small)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

# Metrics
def compute_metrics(pred):
    if hasattr(pred, "predictions") and hasattr(pred, "label_ids"):
        logits = pred.predictions
        labels_ = pred.label_ids
    else:
        logits, labels_ = pred

    preds = np.argmax(logits, axis=-1)
    accuracy = skm.accuracy_score(labels_, preds)
    precision = skm.precision_score(labels_, preds, average="weighted", zero_division=0)
    recall = skm.recall_score(labels_, preds, average="weighted", zero_division=0)
    f1 = skm.f1_score(labels_, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    fp16=False,
    weight_decay=0.01,
)

# snippet to compute class weights and use WeightedTrainer
# assume y_labels list or numpy array of ints for training set
y_labels = np.asarray(train_ds["labels"], dtype=np.int64)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_labels), y=y_labels)
# convert to list
class_weights = class_weights.tolist()

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
)

print("Starting training (this will be CPU-bound and may take some minutes)...")
trainer.train()

# Save model & tokenizer & label maps
print("Saving model to", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w") as f:
    json.dump(label2id, f)
with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w") as f:
    json.dump(id2label, f)

print("Training complete. Model & label maps saved.")
