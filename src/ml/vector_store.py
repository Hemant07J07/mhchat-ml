import json
import os
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data"))
INDEX_FILE = DATA_DIR / "index.json"
FAISS_INDEX_FILE = DATA_DIR / "faiss.index"
FAISS_META_FILE = DATA_DIR / "faiss_meta.json"


def get_retrieval_backend() -> str:
    return os.getenv("RETRIEVAL_BACKEND", "json").strip().lower()


def is_faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


def load_json_index():
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def _to_unit_norm(arr: np.ndarray):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


def build_faiss_index(items):
    import faiss

    embeddings = []
    metadata = []

    for item in items:
        emb = item.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        embeddings.append(emb)
        metadata.append(
            {
                "id": item.get("id"),
                "source": item.get("source", ""),
                "chunk": item.get("chunk", ""),
            }
        )

    if not embeddings:
        raise RuntimeError("No embeddings available to build FAISS index")

    vectors = np.array(embeddings, dtype=np.float32)
    vectors = _to_unit_norm(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    FAISS_META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_faiss_assets():
    import faiss

    if not FAISS_INDEX_FILE.exists() or not FAISS_META_FILE.exists():
        raise FileNotFoundError("FAISS index files not found. Run ingest.py first.")

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    metadata = json.loads(FAISS_META_FILE.read_text(encoding="utf-8"))
    return index, metadata


def search_faiss(query_embedding, top_k: int = 4):
    index, metadata = _load_faiss_assets()

    q = np.array([query_embedding], dtype=np.float32)
    q = _to_unit_norm(q)

    distances, indices = index.search(q, top_k)

    rows = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if float(score) <= 0:
            continue
        rows.append((float(score), metadata[idx]))

    rows.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in rows]
