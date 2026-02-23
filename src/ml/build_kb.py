import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer


def _try_get_faiss():
    try:
        import faiss  # type: ignore
    except Exception as e:
        return None, f"import failed: {e}"

    # On Windows, `pip install faiss` may install a different package that does
    # not expose the real FAISS Python API.
    if not hasattr(faiss, "IndexFlatL2"):
        return None, "module missing IndexFlatL2 (likely not the FAISS package)"

    return faiss, None


def _try_get_annoy():
    try:
        from annoy import AnnoyIndex  # type: ignore

        return AnnoyIndex, None
    except Exception as e:
        return None, f"import failed: {e}"

MODEL = "all-MiniLM-L6-v2"
OUT_DIR = "src/models/kb"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading embedder:", MODEL)
embedder = SentenceTransformer(MODEL)

# Example KB - expand this list with clinical-reviewed content
docs = [
    "Grounding exercise: 5-4-3-2-1. Name 5 things you can see, 4 things you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
    "Diaphragmatic breathing: inhale for 4 seconds, hold 1 second, exhale for 6 seconds. Repeat for 2–5 minutes.",
    "If you are in immediate danger call your local emergency number or contact someone you trust right now.",
    "When anxious, try progressive muscle relaxation: tense each muscle group for 5 seconds and then release.",
    "You’re not alone. If it helps, reach out to a trusted friend or health professional.",
    "Short grounding movement: stomping your feet on the floor for 30 seconds to reconnect with your body.",
    "Use a 5-minute distraction: make tea, doodle, or step outside briefly to change your environment.",
]
print("Encoding docs...")
embs = embedder.encode(docs, convert_to_numpy=True)

faiss, faiss_err = _try_get_faiss()
AnnoyIndex, annoy_err = _try_get_annoy()

if faiss is not None:
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    faiss.write_index(index, f"{OUT_DIR}/kb.index")
    print("FAISS index saved to", f"{OUT_DIR}/kb.index")
elif AnnoyIndex is not None:
    # Annoy fallback
    f = embs.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i, v in enumerate(embs):
        t.add_item(i, v.tolist())
    t.build(10)
    t.save(f"{OUT_DIR}/kb.ann")
    print("Annoy index saved to", f"{OUT_DIR}/kb.ann")
else:
    # Pure NumPy fallback: save embeddings and do brute-force similarity at query time.
    # This avoids native builds (Annoy) and platform limitations (FAISS on Windows).
    np.save(f"{OUT_DIR}/kb_embs.npy", embs)
    print("No FAISS/Annoy available; saved embeddings to", f"{OUT_DIR}/kb_embs.npy")
    print("FAISS reason:", faiss_err)
    print("Annoy reason:", annoy_err)

with open(f"{OUT_DIR}/kb_docs.pkl", "wb") as f:
    pickle.dump(docs, f)
print("Saved docs to", f"{OUT_DIR}/kb_docs.pkl")
print("KB build complete.")
