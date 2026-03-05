from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import numpy as np

KB_DOCS_PATH = "src/models/kb/kb_docs.pkl"
with open(KB_DOCS_PATH, "rb") as f:
    docs = pickle.load(f)
tokenized = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, top_k=10, rerank_k=5):
    # 1) BM25 top candidates
    tok = query.split()
    bm25_scores = bm25.get_scores(tok)
    top_idx = np.argsort(bm25_scores)[::-1][:top_k]
    candidates = [docs[i] for i in top_idx]

    # 2) semantic filter (embedding similarity) - optional
    q_emb = embed_model.encode([query])
    cand_embs = embed_model.encode(candidates)
    # cosine similarity
    sims = (cand_embs @ q_emb.T).squeeze()
    sem_sorted = np.argsort(sims)[::-1]
    # pick top N for rerank combining both signals maybe
    rerank_candidates = [candidates[i] for i in sem_sorted[:rerank_k]]

    # 3) cross-encoder rerank on (query, candidate) pairs
    pairs = [[query, c] for c in rerank_candidates]
    rerank_scores = cross_encoder.predict(pairs)
    order = np.argsort(rerank_scores)[::-1]
    final = [rerank_candidates[i] for i in order]
    return final[:3]