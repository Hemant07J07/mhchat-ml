import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import faiss
    use_faiss = True
except Exception:
    from annoy import AnnoyIndex
    use_faiss = False

MODEL = "all-MiniLM-L6-v2"
KB_DIR = "src/models/kb"

embedder = SentenceTransformer(MODEL)
with open(f"{KB_DIR}/kb_docs.pkl", "rb") as f:
    kb_docs = pickle.load(f)

if use_faiss:
    index = faiss.read_index(f"{KB_DIR}/kb.index")
else:
    dims = embedder.get_sentence_embedding_dimension()
    index = AnnoyIndex(dims, 'angular')
    index.load(f"{KB_DIR}/kb.ann")

def retrieve(query, k=3):
    qv = embedder.encode([query], convert_to_numpy=True)
    if use_faiss:
        D, I = index.search(qv, k)
        idxs = I[0].tolist()
    else:
        idxs = index.get_nns_by_vector(qv[0].tolist(), k, include_distance=False)
    return [kb_docs[i] for i in idxs]

# quick mannual test
if __name__ == "__main__":
    queries = [
        "I feel anxious and can't stop shaking",
        "How fo I do breathing exercises?",
        "I think I'm going to hurt myself"
    ]
    for q in queries:
        print("QUERY:", q)
        hits = retrieve(q, k=3)
        for i, h in enumerate(hits, 1):
            print(f"Hit {i}:", h)
        print("-"*60)