import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import uvicorn
from src.ml.fast_infer import predict as intent_predict
from src.ml.safety_checks import contains_crisis_words
from src.ml.retrieve_kb import retrieve as kb_retrieve
from src.ml.rag_answer import chat as rag_chat
from src.ml.rag_answer import chat_stream as rag_chat_stream

logger = logging.getLogger(__name__)

app = FastAPI(title="mhchat-ml-api")

class PredictRequest(BaseModel):
    message: str
    context: Optional[List[dict]] = None  # Previous messages: [{"sender": "user|bot", "text": "..."}, ...]

class PredictResponse(BaseModel):
    intent: str
    intent_score: float
    crisis: bool
    kb_hits: List[str]


class ChatRequest(BaseModel):
    message: str
    context: Optional[List[dict]] = None
    fallback_hint: Optional[str] = None


class ChatResponse(BaseModel):
    intent: str
    intent_score: float
    crisis: bool
    reply: str
    summary: str
    web_highlights: List[str]
    sources: List[str]
    kb_hits: List[str]

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring service availability."""
    return {"status": "healthy", "service": "mhchat-ml-api", "version": "1.0"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict intent, crisis flag, and retrieve relevant KB documents.
    
    Logic:
    1. Detect crisis keywords and use crisis intent classifier
    2. Classify intent (casual_chat, mental_health_support, crisis)
    3. If casual_chat: skip KB retrieval (no relevant docs for greetings)
    4. If mental_health_support: retrieve relevant KB documents with threshold
    5. Return intent, score, crisis flag, and KB hits
    """
    text = req.message or ""
    context = req.context or []
    
    logger.info(f"Prediction request: '{text[:50]}...'")
    
    # 1) Safety keyword check (detects crisis keywords)
    kw_flag = contains_crisis_words(text)
    
    # 2) Intent classification
    r = intent_predict(text)  # returns {"label":..., "score":..., "all_probs": [...]}
    intent_label = r.get("label", "casual_chat")
    intent_score = float(r.get("score", 0.0))
    
    logger.info(f"Intent: {intent_label} ({intent_score:.3f}), Crisis keywords: {kw_flag}")
    
    # 3) Crisis decision: keyword flag OR high-confidence crisis intent
    crisis_flag = kw_flag or (intent_label == "crisis" and intent_score > 0.6)
    
    # 4) KB retrieval with intent-based gating
    kb_hits = []
    
    if crisis_flag:
        # Crisis detected: return emergency resource
        kb_hits = ["If you are in immediate danger, call your local emergency number or a crisis hotline right now."]
        logger.info("Crisis detected - returning emergency resource")
    
    elif intent_label == "casual_chat" or intent_score < 0.5:
        # Casual chat or low-confidence intent: skip KB retrieval
        # Don't force mental health advice for "hello", "what is this?", etc.
        kb_hits = []
        logger.info(f"Casual chat detected (intent={intent_label}, score={intent_score:.3f}) - skipping KB")
    
    elif intent_label == "mental_health_support":
        # Mental health query: retrieve relevant docs with relevance threshold
        # threshold=1.0 means more lenient filtering for L2 distance
        # (L2 distance ranges from 0 for identical to ~1.4 for orthogonal vectors)
        kb_hits = kb_retrieve(text, k=3, threshold=1.0)
        logger.info(f"Mental health query - retrieved {len(kb_hits)} KB documents")
        
        if not kb_hits:
            logger.debug(f"No KB documents met threshold for: '{text[:50]}...'")
    
    else:
        logger.info(f"Unknown intent: {intent_label}")
        kb_hits = []
    
    return PredictResponse(
        intent=intent_label,
        intent_score=round(intent_score, 3),
        crisis=crisis_flag,
        kb_hits=kb_hits
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message or ""
    fallback_hint = req.fallback_hint

    logger.info(f"Chat request: '{text[:50]}...'")

    kw_flag = contains_crisis_words(text)
    r = intent_predict(text)
    intent_label = r.get("label", "casual_chat")
    intent_score = float(r.get("score", 0.0))

    crisis_flag = kw_flag or (intent_label == "crisis" and intent_score > 0.6)

    kb_hits = []
    if crisis_flag:
        kb_hits = [
            "If you are in immediate danger, call your local emergency number or a crisis hotline right now."
        ]
        reply = kb_hits[0]
        return ChatResponse(
            intent=intent_label,
            intent_score=round(intent_score, 3),
            crisis=crisis_flag,
            reply=reply,
            summary="",
            web_highlights=[],
            sources=[],
            kb_hits=kb_hits,
        )

    if intent_label == "mental_health_support" and intent_score >= 0.5:
        kb_hits = kb_retrieve(text, k=3, threshold=1.0)

    rag_result = rag_chat(text, kb_hits, fallback_hint)
    return ChatResponse(
        intent=intent_label,
        intent_score=round(intent_score, 3),
        crisis=crisis_flag,
        reply=rag_result.get("answer", ""),
        summary=rag_result.get("summary", ""),
        web_highlights=rag_result.get("web_highlights", []),
        sources=rag_result.get("sources", []),
        kb_hits=kb_hits,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    text = req.message or ""
    fallback_hint = req.fallback_hint

    logger.info(f"Chat stream request: '{text[:50]}...'")

    kw_flag = contains_crisis_words(text)
    r = intent_predict(text)
    intent_label = r.get("label", "casual_chat")
    intent_score = float(r.get("score", 0.0))

    crisis_flag = kw_flag or (intent_label == "crisis" and intent_score > 0.6)

    kb_hits = []
    if crisis_flag:
        kb_hits = [
            "If you are in immediate danger, call your local emergency number or a crisis hotline right now."
        ]
        payload = {
            "intent": intent_label,
            "intent_score": round(intent_score, 3),
            "crisis": crisis_flag,
            "reply": kb_hits[0],
            "summary": "",
            "web_highlights": [],
            "sources": [],
            "kb_hits": kb_hits,
        }

        def _single():
            yield json.dumps({"type": "final", "data": payload}) + "\n"

        return StreamingResponse(_single(), media_type="application/x-ndjson")

    if intent_label == "mental_health_support" and intent_score >= 0.5:
        kb_hits = kb_retrieve(text, k=3, threshold=1.0)

    def _stream():
        for event in rag_chat_stream(text, kb_hits, fallback_hint):
            yield json.dumps(event) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
    