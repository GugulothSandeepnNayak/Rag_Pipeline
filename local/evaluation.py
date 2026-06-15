from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")


def bert_score_like(pred, ref):
    """A simple BERT-like score using sentence-transformers cosine similarity.

    Returns a float in [-1, 1] (higher is more similar).
    """
    emb1 = model.encode([pred])
    emb2 = model.encode([ref])

    return float(cosine_similarity(emb1, emb2)[0][0])


def tokenize(text):
    return text.lower().split()


def ngrams(tokens, n):
    if n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def bleu_score(pred, ref, max_n=4, smooth=1e-9):
    """Compute a simple cumulative BLEU score (n=1..max_n) with brevity penalty.

    Returns a float in [0,1]. This is a lightweight implementation and not a
    drop-in replacement for sacrebleu/nltk BLEU but works without extra deps.
    """
    p_tokens = tokenize(pred)
    r_tokens = tokenize(ref)

    if len(p_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n+1):
        p_ngrams = ngrams(p_tokens, n)
        r_ngrams = ngrams(r_tokens, n)

        if len(p_ngrams) == 0:
            precisions.append(0.0)
            continue

        # count matches using multiset semantics
        from collections import Counter

        r_counts = Counter(r_ngrams)
        matches = 0
        for ng in p_ngrams:
            if r_counts.get(ng, 0) > 0:
                matches += 1
                r_counts[ng] -= 1

        precisions.append(matches / len(p_ngrams))

    # geometric mean of precisions
    import math

    # smoothing to avoid log(0)
    log_sum = 0.0
    for p in precisions:
        log_sum += math.log(p + smooth)

    geo_mean = math.exp(log_sum / max_n)

    # brevity penalty
    ref_len = len(r_tokens)
    pred_len = len(p_tokens)
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)

    return float(bp * geo_mean)


def precision(pred, ref):
    """Token-level precision: overlap / number of predicted tokens."""
    p_tokens = tokenize(pred)
    r_tokens = tokenize(ref)

    if len(p_tokens) == 0:
        return 0.0

    from collections import Counter

    r_counts = Counter(r_tokens)
    overlap = 0
    for t in p_tokens:
        if r_counts.get(t, 0) > 0:
            overlap += 1
            r_counts[t] -= 1

    return float(overlap / len(p_tokens))


def llm_score(pred, ref, model_name="llama3"):
    """Ask a local LLM (via ollama) to rate the predicted answer against the
    reference on a 0.0-1.0 scale. Requires `ollama` package and running daemon.

    Returns a float in [0,1] or raises an informative exception if unavailable.
    """
    try:
        import ollama
    except Exception as e:
        raise RuntimeError("ollama is required for llm_score; install and run the Ollama daemon") from e

    prompt = (
        "You are an evaluator. Given a reference answer and a predicted answer, "
        "rate how well the predicted answer matches the reference on a scale 0.0 to 1.0. "
        "1.0 means the prediction is fully correct and faithful; 0.0 means unrelated or wrong. "
        "Output only a single floating point number between 0 and 1.\n\n"
        f"Reference: {ref}\n"
        f"Prediction: {pred}\n"
        "Score:"
    )

    # Note: older/newer versions of the `ollama` client may not accept a
    # `timeout` keyword on `chat()`; call with the standard args only.
    try:
        resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    except TypeError:
        # Some versions expose a `Client` object; try the top-level function first
        try:
            client = ollama.Ollama()
            resp = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            raise RuntimeError(f"Failed to call Ollama chat API: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call Ollama chat API: {e}") from e

    content = resp.get("message", {}).get("content", "").strip()
    # attempt to parse a float from the response
    import re

    m = re.search(r"([0-1](?:\.\d+)?|0?\.\d+)", content)
    if not m:
        raise ValueError(f"Could not parse numeric score from LLM response: {content!r}")

    val = float(m.group(1))
    # clamp
    return max(0.0, min(1.0, val))


def evaluate(pred, ref, use_llm=False, llm_model="llama3"):
    """Run all evaluation metrics and return a dict of scores.

    If `use_llm` is True, `llm_score` will be invoked (requires Ollama).
    """
    results = {
        "bert_sim": bert_score_like(pred, ref),
        "bleu": bleu_score(pred, ref),
        "precision": precision(pred, ref),
    }

    if use_llm:
        try:
            results["llm_score"] = llm_score(pred, ref, model_name=llm_model)
        except Exception as e:
            results["llm_score_error"] = str(e)

    return results