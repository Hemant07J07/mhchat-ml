import os
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import sklearn.metrics as skm

# Config - tuned for CPU / low RAM
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 8       # small to fit RAM
NUM_EPOCHS = 2       # keep low for quick runs
OUTPUT_DIR = "src/models/intent"

parser = argparse.ArgumentParser(description="Train intent classifier")
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
    # `transformers` may pass either an EvalPrediction-like object
    # (with .predictions / .label_ids) or a (predictions, labels) tuple.
    if hasattr(pred, "predictions") and hasattr(pred, "label_ids"):
        logits = pred.predictions
        labels = pred.label_ids
    else:
        logits, labels = pred

    preds = np.argmax(logits, axis=-1)
    accuracy = skm.accuracy_score(labels, preds)
    precision = skm.precision_score(labels, preds, average="weighted", zero_division=0)
    recall = skm.recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = skm.f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1":f1}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    fp16=False,      # CPU:don't use fp16
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
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