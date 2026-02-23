
# mhchat-ml

Minimal ML + API package for a mental-health chat prototype:

- **Intent classifier** (HF Transformers) trained on a small CSV dataset.
- **Safety keyword check** for crisis phrases.
- **Tiny knowledge base (KB)** retrieval (FAISS / Annoy fallback) for non-crisis replies.
- **FastAPI** endpoint `POST /predict` that returns intent + crisis flag + KB hits.

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

