import re
from typing import List, Tuple

BAD_WORDS = ["idiot", "stupid", "hate"]


def is_toxic(query: str) -> bool:
    """Detect toxic/hateful language using simple word-boundary checks.

    This is intentionally lightweight. For production use a moderated list
    or third-party moderation API.
    """
    if not query:
        return False

    text = query.lower()
    for word in BAD_WORDS:
        # match whole words only
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return True
    return False


def low_confidence(scores, threshold=0.2):
    """Return True when the top retrieval score is below a threshold.

    `scores` is expected to be a sequence where higher values mean better
    matches (e.g., similarity). Default threshold 0.2 is conservative.
    """
    try:
        if not scores:
            return True
        return float(scores[0]) < float(threshold)
    except Exception:
        return False


def semantic_grounding_score(answer: str, contexts: List[str]) -> Tuple[float, int, int]:
    """Compute a semantic grounding score between sentences in `answer`
    and the list of `contexts` (chunks).

    Returns a tuple `(best_score, best_context_index, best_answer_sentence_index)`.
    The score is the maximum cosine similarity between any answer sentence and
    any context chunk using the sentence-transformers model found in
    `evaluation.model`.

    If no content is provided, returns `(0.0, -1, -1)`.
    """
    if not answer or not contexts:
        return 0.0, -1, -1

    # Lazy import to avoid circular imports at module load
    try:
        from evaluation import model
    except Exception:
        return 0.0, -1, -1

    # split answer into sentences (simple split on punctuation/newline)
    ans_sents = [s.strip() for s in re.split(r'[\.\?!\n]', answer) if s.strip()]
    if not ans_sents:
        return 0.0, -1, -1

    # encode all answer sentences and all contexts in one batch each
    try:
        ans_emb = model.encode(ans_sents, convert_to_numpy=True)
        ctx_emb = model.encode(contexts, convert_to_numpy=True)
    except Exception:
        # fallback: return zero if encoding fails
        return 0.0, -1, -1

    # compute pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(ans_emb, ctx_emb)  # shape (n_ans, n_ctx)
    # find global max
    import numpy as np

    max_idx = np.unravel_index(np.argmax(sims), sims.shape)
    best_ans_i, best_ctx_i = int(max_idx[0]), int(max_idx[1])
    best_score = float(sims[best_ans_i, best_ctx_i])

    return best_score, best_ctx_i, best_ans_i


def validate_answer(answer: str, context: str) -> bool:
    """Backward-compatible literal grounding check (kept for compatibility).

    Prefer `semantic_grounding_score()` for better grounding detection.
    """
    if not answer or not context:
        return False

    ctx = context.lower()
    for sentence in answer.split('.'):
        s = sentence.strip().lower()
        if not s:
            continue
        if s in ctx:
            return True
    return False