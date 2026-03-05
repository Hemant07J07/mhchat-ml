import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import json

MODEL_PATH = "onnx/model.onnx"
tokenizer = AutoTokenizer.from_pretrained("src/models/intent")
sess = ort.InferenceSession(MODEL_PATH)

def predict_onnx(text):
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="np")
    inputs = {k: v for k,v in enc.items()}
    # ONNX input names depend on the exported model; check sess.get_inputs()
    out = sess.run(None, inputs)[0]  # logits
    probs = np.exp(out) / np.exp(out).sum(axis=-1, keepdims=True)
    best = int(np.argmax(probs, axis=-1).item())
    return {"label_idx": best, "score": float(probs[0,best]), "all_probs": probs[0].tolist()}
