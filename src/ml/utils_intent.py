import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_intent_pipeline(model_path="src/models/intent"):
    """
    Returns a Hugging Face pipeline for text-classification and id2label map.
    """
    # load label map
    try:
        with open(f"{model_path}/id2label.json", "r") as f:
            id2label = json.load(f)
    except Exception:
        id2label = None

    # pipeline (this will load model & tokenizer)
    nlp = pipeline("text-classification", model=model_path, tokenizer=model_path, return_all_scores=True)
    return nlp, id2label
