import os
from functools import lru_cache

from sentence_transformers import CrossEncoder

# Small cross-encoder for similarity-style scoring against crisis templates.
# NOTE: previous id had a typo; the public model is `ms-marco`.
MODEL = os.environ.get("CRISIS_CROSSENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    # If you have a bad HF token set in your environment, requests can fail with 401.
    # `token=False` explicitly disables auth headers and works for public models.
    local_only = os.environ.get("CRISIS_CE_LOCAL_ONLY", "0") == "1"
    return CrossEncoder(
        MODEL,
        device=os.environ.get("CRISIS_CE_DEVICE", None),
        local_files_only=local_only,
        token=False,
    )

def crisis_score(text):
    """
    Return a score (higher -> more likely crisis).
    Tune the threshold on a validation set.
    """
    ce = _get_cross_encoder()

    # CrossEncoder expects list of [text, template] pairs.
    # We score similarity to crisis templates and take max score.
    crisis_templates = [
        "I want to end my life",
        "I will kill myself",
        "I have a plan to harm myself",
        "I am going to hurt myself"
    ]
    pairs = [[text, t] for t in crisis_templates]
    scores = ce.predict(pairs) # shape (len(templates),)
    # return max score
    return float(max(scores))


if __name__ == "__main__":
    sample = "I feel hopeless and I want to end my life"
    print("model:", MODEL)
    print("score:", crisis_score(sample))