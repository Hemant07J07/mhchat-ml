---
title: mhchat-ml
sdk: docker
app_port: 7860
---

# mhchat-ml

Minimal ML + API package for a mental-health chat prototype:

- **Intent classifier** (HF Transformers) trained on a small CSV dataset.
- **Safety keyword check** for crisis phrases.
- **Tiny knowledge base (KB)** retrieval (FAISS / Annoy fallback) for non-crisis replies.
- **FastAPI** endpoint `POST /predict` that returns intent + crisis flag + KB hits.
- **RAG chat** endpoint `POST /chat` that can use Ollama, local docs, and web search.
- **Streaming RAG** endpoint `POST /chat/stream` that emits JSONL events.

## High-level flow

1) Safety keywords are checked first.
2) Intent classification runs to decide whether to retrieve KB docs.
3) Local retrieval (embeddings) and web search gather context.
4) Ollama generates a structured JSON response combining local + web.
5) If Ollama is unavailable, a synthesized fallback is returned.

## Repo Layout

- `data/`
	- `intent_data.csv`: base training data (`text,label`)
	- `intent_data_expanded.csv`: synthetic expansion output
	- `intent_data_oversampled.csv`: oversampled output
- `src/ml/`
	- `train_intent.py`: train intent model from a CSV
	- `eval_intent.py`: evaluate intent model on a held-out split
	- `infer_intent.py`: simple local inference demo (prints predictions)
	- `fast_infer.py`: lightweight inference helper used by the API
	- `safety_checks.py`: keyword-based crisis detection
	- `build_kb.py`: build a vector index for KB docs
	- `retrieve_kb.py`: query the KB index
	- `rag_answer.py`: Ollama-powered RAG answers (with streaming helper)
	- `rag_retrieval.py`: local + web retrieval helpers
	- `mcp_client.py`: optional MCP tool wrapper with local fallback
	- `expand_dataset.py`: generate synthetic intent examples
	- `oversample_intent.py`: oversample crisis class in the dataset
- `src/models/`
	- `intent/`: trained model artifacts + label maps (`id2label.json`, `label2id.json`)
	- `kb/`: KB artifacts (`kb.index`, `kb_docs.pkl`, etc.)
- `src/api/main.py`: FastAPI app
- `tests/env_check.py`: quick environment sanity check

## Setup (Windows)

From the repo root (`C:\Users\...\mhchat-ml`):

```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

# Optional: RAG + Ollama dependencies
pip install -r requirements-rag.txt
```

Optional: to avoid HF download warnings / rate limits, set an auth token:

```powershell
$env:HF_TOKEN = "<your_huggingface_token>"
```

Quick environment check:

```powershell
python tests\env_check.py
```

## Build the KB (one-time)

This embeds a small list of KB documents and writes an index under `src/models/kb/`.

```powershell
python src\ml\build_kb.py
```

Outputs (depending on what’s available on your machine):

- `src/models/kb/kb.index` (FAISS)
- or `src/models/kb/kb.ann` (Annoy)
- or `src/models/kb/kb_embs.npy` (pure NumPy fallback)
- plus `src/models/kb/kb_docs.pkl` (the KB strings)

## Intent Model: dataset utilities

### 1) Expand the dataset (synthetic templates)

Generates `data/intent_data_expanded.csv`.

```powershell
python src\ml\expand_dataset.py
```

### 2) Oversample the crisis class

Reads `data/intent_data.csv` and writes `data/intent_data_oversampled.csv`.

```powershell
python src\ml\oversample_intent.py
```

## Train the intent classifier

Default training uses `distilbert-base-uncased` and saves artifacts to `src/models/intent/`.

Train on the base dataset:

```powershell
python src\ml\train_intent.py --data-file data\intent_data.csv
```

Train on the oversampled dataset:

```powershell
python src\ml\train_intent.py --data-file data\intent_data_oversampled.csv
```

Train on the expanded dataset:

```powershell
python src\ml\train_intent.py --data-file data\intent_data_expanded.csv
```

## Evaluate the intent model

Runs a held-out split on `data/intent_data_oversampled.csv` and prints a classification report + confusion matrix.

```powershell
python src\ml\eval_intent.py
```

## Local inference demos

### Intent inference demo (prints predictions)

```powershell
python src\ml\infer_intent.py
```

### KB retrieval demo (prints top hits)

```powershell
python src\ml\retrieve_kb.py
```

## Run the API

Start FastAPI with Uvicorn:

```powershell
python -m uvicorn src.api.main:app --reload --port 8001
```

Open docs:

- http://127.0.0.1:8001/docs

## RAG configuration

The RAG flow uses Ollama for embeddings and chat completions, plus optional web search
and MCP tool integration. Configure via environment variables:

```env
# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:1b
OLLAMA_EMBED_MODEL=embeddinggemma
OLLAMA_FALLBACK_CHAT_MODEL=llama3.2:1b
MODEL_PROFILE=LLAMA3_1B

# Retrieval
RAG_DATA_DIR=data
RETRIEVAL_BACKEND=json
LOCAL_TOP_K=3
WEB_TOP_K=2

# MCP (optional)
MCP_ENABLED=false
MCP_SERVER_PATH=mcp_server.py
```

## Deployment (Hugging Face Docker Space)

This repo is ready for a Docker Space. The container listens on port 7860.

1) Create a new Space, choose Docker.
2) Push this repo to the Space.
3) The Space will build using the Dockerfile and run:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 7860
```

Your live URL will look like: https://<your-space>.hf.space

## RAG endpoints

### `/chat`

Returns a structured answer with sources.

```powershell
$body = @{ message = 'How do I handle panic attacks?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/chat' -ContentType 'application/json' -Body $body
```

### `/chat/stream`

Streams JSONL events with tokens and a final payload. Each line is a JSON object:

- `{ "type": "token", "value": "..." }`
- `{ "type": "final", "data": { ... } }`

```bash
curl -N -X POST "http://127.0.0.1:8001/chat/stream" \
	-H "Content-Type: application/json" \
	-d "{\"message\":\"What are grounding exercises?\"}"
```

### Test with PowerShell

```powershell
$body = @{ message = 'I want to kill myself' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/predict' -ContentType 'application/json' -Body $body

$body = @{ message = 'How to do breathing exercises' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/predict' -ContentType 'application/json' -Body $body
```

### Test with curl

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
	-H "Content-Type: application/json" \
	-d "{\"message\":\"I want to kill myself\"}"

curl -X POST "http://127.0.0.1:8001/predict" \
	-H "Content-Type: application/json" \
	-d "{\"message\":\"How to do breathing exercises\"}"
```

Expected response shape:

```json
{
	"intent": "crisis",
	"intent_score": 0.94,
	"crisis": true,
	"kb_hits": ["If you are in immediate danger..."]
}
```

## How `/predict` works

1. `src/ml/safety_checks.py` checks for crisis keywords.
2. `src/ml/fast_infer.py` runs the intent classifier and returns `{label, score, all_probs}`.
3. If crisis is detected, the API returns an emergency message and skips KB retrieval.
4. Otherwise, `src/ml/retrieve_kb.py` fetches top-K KB snippets.

## Troubleshooting

- **First run downloads models** (intent base model and/or sentence-transformer). This can take a minute.
- **Annoy on Windows**: `annoy` is a native extension and may require Microsoft C++ Build Tools to compile.
	- This repo pins `annoy` only for Python `< 3.13` in `requirements.txt`.
- **FAISS on Windows**: make sure you installed the real `faiss-cpu` package; some similarly named packages don’t expose the full API.

