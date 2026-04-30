import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

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

def retrieve(query, k=3, threshold=0.6):
    """
    Retrieve KB documents similar to query using semantic embeddings.
    
    Args:
        query: User query string
        k: Maximum documents to retrieve
        threshold: Relevance threshold (0-1). For FAISS L2: lower=more similar.
                 For Annoy angular: lower=more similar. 
                 Only return docs below this distance threshold.
    
    Returns:
        List of relevant KB documents (filtered by threshold)
    """
    qv = embedder.encode([query], convert_to_numpy=True)
    
    if use_faiss:
        # FAISS L2 distance: lower = more similar
        D, I = index.search(qv, k)
        idxs = I[0].tolist()
        distances = D[0].tolist()
        
        # Convert FAISS L2 distance to similarity score (0-1)
        # L2 distance scaled to ~0-2 range for normalized embeddings
        # threshold of 0.6 roughly means "moderately similar"
        filtered_hits = []
        for idx, dist in zip(idxs, distances):
            # Lower distance = higher similarity
            # For normalized embeddings, L2 distance of 0 = identical, ~1.4 = orthogonal
            if dist < threshold:
                filtered_hits.append(kb_docs[idx])
                logger.debug(f"KB hit: dist={dist:.3f}, doc='{kb_docs[idx][:50]}...'")
        
        return filtered_hits if filtered_hits else []
    else:
        # Annoy angular distance (0-2 range)
        # Angular distance is in range [0, 2] where:
        # 0 = identical direction, 2 = opposite direction
        idxs = index.get_nns_by_vector(qv[0].tolist(), k, include_distance=True)
        hits_with_dist = idxs  # Returns list of (idx, distance) tuples
        
        filtered_hits = []
        for item in hits_with_dist:
            if isinstance(item, tuple):
                idx, dist = item
            else:
                # Fallback if format differs
                idx = item
                dist = 0
            
            if dist < threshold:
                filtered_hits.append(kb_docs[idx])
                logger.debug(f"KB hit: dist={dist:.3f}, doc='{kb_docs[idx][:50]}...'")
        
        return filtered_hits if filtered_hits else []

# quick manual test
if __name__ == "__main__":
    queries = [
        "I feel anxious and can't stop shaking",
        "How do I do breathing exercises?",
        "I think I'm going to hurt myself",
        "hello",
        "what is this?"
    ]
    for q in queries:
        print("QUERY:", q)
        hits = retrieve(q, k=3, threshold=0.6)
        if hits:
            for i, h in enumerate(hits, 1):
                print(f"Hit {i}:", h[:80])
        else:
            print("(No relevant KB documents found)")
        print("-"*60)