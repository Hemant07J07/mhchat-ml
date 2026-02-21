import sys
print("Python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch import error:", e)

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers import error:", e)

try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers OK")
except Exception as e:
    print("sentence-transformers import error:", e)

# test small embedding (this will download a small model the first time)
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    v = m.encode("hello world")
    print("embedding vector length:", len(v))
except Exception as e:
    print("embedding test error (might download model):", e)