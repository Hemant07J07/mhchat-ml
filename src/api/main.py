from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import uvicorn
from src.ml.fast_infer import predict as intent_predict
from src.ml.safety_checks import contains_crisis_words
from src.ml.retrieve_kb import retrieve as kb_retrieve

app = FastAPI(title="mhchat-ml-api")

class PredictRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    intent: str
    intent_score: float
    crisis: bool
    kb_hits: List[str]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.message or ""
    # 1) safety keyword check
    kw_flag = contains_crisis_words(text)

    # 2) intent model
    r = intent_predict(text) # returns {"label":..., "score":..., "all_probs": [...]}
    intent_label = r.get("label")
    intent_score = float(r.get("score", 0.0))

    # 3) crisis decision: either rule or model predicts crisis with high prob
    crisis_flag = kw_flag or (intent_label == "crisis") or (intent_score > 0.7 and intent_label == "crisis")

    # 4) kb retrieve (skip if crisis true - show emergency resource insted)
    if crisis_flag:
        kb_hits = ["If you are in immediate danger, call your local emergency number or a crisis hotline right now."]
    else:
        kb_hits = kb_retrieve(text, k=3)

    return PredictResponse(intent=intent_label, intent_score=round(intent_score, 3), crisis=crisis_flag, kb_hits=kb_hits)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
    