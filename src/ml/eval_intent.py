import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "src/models/intent"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
with open(f"{MODEL_PATH}/id2label.json") as f:
    id2label = json.load(f)
with open(f"{MODEL_PATH}/label2id.json") as f:
    label2id = json.load(f)
inv_map = {int(k):v for k,v in id2label.items()}

ds = load_dataset("csv", data_files="data/intent_data_oversampled.csv", split="train")
ds = ds.train_test_split(test_size=0.15, seed=42)["test"]

y_true = []
y_pred = []
for ex in ds:
    txt = ex["text"]
    enc = tokenizer(txt, truncation=True, padding=True, return_tensors="pt", max_length=64)
    with torch.no_grad():
        logits = model(**enc).logits
    pred = int(torch.argmax(logits).item())
    y_pred.append(pred)
    y_true.append(label2id[ex["label"]])

print(classification_report(y_true, y_pred, target_names=[inv_map[i] for i in sorted(inv_map.keys())]))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))