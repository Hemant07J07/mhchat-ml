import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

MODEL_PATH = "src/models/intent"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

with open(f"{MODEL_PATH}/id2label.json") as f:
    id2label = json.load(f)

def predict(text):
    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=64)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    best_idx = int(torch.argmax(logits).item())
    best_label = id2label.get(str(best_idx), f"LABEL_{best_idx}")
    return {"label": best_label, "score": probs[best_idx], "all_probs": probs}
