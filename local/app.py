import streamlit as st
from rag_pipeline import retrieve, generate_answer
from evaluation import evaluate
import guardrails

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot — Improved Streamlit UI")

# 🔥 Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Retrieval tuning controls (sidebar)
with st.sidebar.expander("Retrieval / Rerank settings", expanded=True):
    use_rerank = st.checkbox("Enable re-ranking (cross-encoder + hybrid)", value=True)
    rerank_k = st.number_input("Candidate pool size (rerank_k)", min_value=5, max_value=500, value=50, step=5)
    final_k = st.number_input("Results to return (k)", min_value=1, max_value=20, value=3)
    alpha = st.slider("Cross-encoder weight (alpha)", 0.0, 1.0, 0.6)
    beta = st.slider("Dense-sim weight (beta)", 0.0, 1.0, 0.3)
    gamma = st.slider("TF-IDF weight (gamma)", 0.0, 1.0, 0.1)
    st.markdown("---")
    st.write("Preset weights:")
    if st.button("Fast (no rerank)"):
        st.session_state.update({"use_rerank": False, "rerank_k": 10, "alpha": 0.0, "beta": 1.0, "gamma": 0.0})
    if st.button("Balanced"):
        st.session_state.update({"use_rerank": True, "rerank_k": 40, "alpha": 0.5, "beta": 0.4, "gamma": 0.1})
    if st.button("High quality"):
        st.session_state.update({"use_rerank": True, "rerank_k": 100, "alpha": 0.7, "beta": 0.2, "gamma": 0.1})

st.sidebar.markdown("---")
st.sidebar.header("Benchmark")
bench_file = st.sidebar.file_uploader("Upload CSV (columns: query, reference)", type=["csv"])
bench_threshold = st.sidebar.slider("Match threshold (bert-sim) for semantic match", 0.0, 1.0, 0.7)
run_benchmark = st.sidebar.button("Run benchmark")

# Layout: left = chat, right = context / evaluation
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Chat")

    # Display chat history in a scrollable container
    chat_container = st.container()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input form (keeps UI tidy)
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("Ask something...", key="query_input")
        expected = st.text_input("Expected answer (optional)", key="expected_input")
        use_llm_eval = st.checkbox("Use LLM to evaluate (requires Ollama)", key="use_llm_eval")
        send = st.form_submit_button("Send")

    if send and query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Guardrail: block toxic queries early
        if guardrails.is_toxic(query):
            refusal = "Sorry — I can't assist with hateful or abusive language."
            st.session_state.messages.append({"role": "assistant", "content": refusal})
            with st.chat_message("assistant"):
                st.write(refusal)
        else:
            # 🔍 Retrieve context
            if use_rerank:
                results, scores, ids = retrieve(query, k=final_k, rerank_k=rerank_k, alpha=alpha, beta=beta, gamma=gamma)
            else:
                # dense baseline
                from rag_pipeline import dense_retrieve

                results, scores, ids = dense_retrieve(query, k=final_k)

            # Log retrieval for debugging
            try:
                with open("retrieval_debug.log", "a", encoding="utf-8") as fh:
                    fh.write("--- New query ---\n")
                    fh.write(f"query: {query}\n")
                    fh.write(f"num_results: {len(results)}\n")
                    for i, r in enumerate(results[:10]):
                        fh.write(f"[{i}] id={ids[i]} score={scores[i] if i < len(scores) else None} len={len(r) if r else 0}\n")
                        fh.write(repr(r)[:1000] + "\n")
                    fh.write("\n")
            except Exception:
                pass

            # 🤖 Generate response
            answer = generate_answer(st.session_state.messages, query, results)

            # Post-generation guardrails: check grounding (semantic) and low confidence
            grounding_score, best_ctx_i, best_ans_i = guardrails.semantic_grounding_score(answer, results)
            # tag low confidence if retrieval scores are low
            if guardrails.low_confidence(scores):
                answer = f"[Low confidence] {answer}"

            # only add the non-grounded warning when semantic grounding is below threshold
            grounding_threshold = 0.78
            is_grounded = grounding_score >= grounding_threshold
            if not is_grounded:
                answer = f"[Warning: answer not grounded in retrieved context] {answer}"

            # If user provided an expected answer, compute evaluation metrics
            eval_results = None
            if expected:
                eval_results = evaluate(answer, expected, use_llm=use_llm_eval)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.write(answer)

            # After generating, surface detailed info in the right column
            with col2:
                st.subheader("Retrieved Context")
                top_k = min(3, len(results))
                top_chunks = results[:top_k]
                top_scores = scores[:top_k]
                top_ids = ids[:top_k]

                with st.expander(f"🔝 Top {top_k} Chunks (used for LLM)"):
                    for i, chunk in enumerate(top_chunks):
                        scr = top_scores[i] if i < len(top_scores) else None
                        sid = top_ids[i] if i < len(top_ids) else None
                        st.write(f"**Top {i+1} — id: {sid} (score: {scr:.3f} if scr else 'N/A')**")
                        st.write(chunk)
                        st.write("---")

                # Show grounding score and best-matching chunk id
                st.markdown("**Grounding**")
                gs_text = f"{grounding_score:.3f}"
                best_id = top_ids[best_ctx_i] if (best_ctx_i is not None and best_ctx_i >= 0 and best_ctx_i < len(ids)) else None
                st.write(f"Semantic grounding score: {gs_text} (best match chunk id: {best_id})")

                with st.expander("📚 All Sources"):
                    for i, chunk in enumerate(results):
                        sid = ids[i] if i < len(ids) else None
                        scr = scores[i] if i < len(scores) else None
                        scr_text = f"{scr:.3f}" if scr is not None else "N/A"
                        st.write(f"**Chunk {i+1} — id: {sid} (score: {scr_text})**")
                        st.write(chunk[:300] + "...")
                        st.write("---")

                if eval_results is not None:
                    with st.expander("🏷️ Evaluation"):
                        accuracy = 1.0 if answer.strip() == expected.strip() else 0.0
                        st.write(f"**Accuracy (exact match):** {accuracy:.2f}")
                        st.write(f"**BERT-sim:** {eval_results.get('bert_sim'):.4f}")
                        st.write(f"**BLEU:** {eval_results.get('bleu'):.4f}")
                        st.write(f"**Precision:** {eval_results.get('precision'):.4f}")
                        if 'llm_score' in eval_results:
                            st.write(f"**LLM score:** {eval_results.get('llm_score'):.4f}")
                        if 'llm_score_error' in eval_results:
                            st.write(f"**LLM error:** {eval_results.get('llm_score_error')}")

with col2:
    st.header("Context & Tools")
    st.write("Use the controls in the sidebar to tune retrieval and run benchmarks.")

# --- Benchmark runner (CSV)
if bench_file is not None and run_benchmark:
    import pandas as _pd
    from evaluation import bert_score_like
    from rag_pipeline import dense_retrieve

    df = _pd.read_csv(bench_file)
    if 'query' not in df.columns or 'reference' not in df.columns:
        st.sidebar.error("CSV must contain 'query' and 'reference' columns")
    else:
        rows = []
        total = len(df)
        progress = st.progress(0)
        for i, row in df.iterrows():
            q = str(row['query'])
            ref = str(row['reference'])

            # baseline (dense)
            base_docs, base_scores, base_ids = dense_retrieve(q, k=final_k)
            # reranked (hybrid) if enabled
            rr_docs, rr_scores, rr_ids = (retrieve(q, k=final_k, rerank_k=rerank_k, alpha=alpha, beta=beta, gamma=gamma)
                                          if use_rerank else (base_docs, base_scores, base_ids))

            # match check: consider a match if reference substring in doc OR bert_sim >= threshold
            def has_match(docs):
                for d in docs:
                    if d and ref.lower() in d.lower():
                        return True
                    try:
                        if bert_score_like(d, ref) >= bench_threshold:
                            return True
                    except Exception:
                        pass
                return False

            base_hit = has_match(base_docs)
            rr_hit = has_match(rr_docs)

            # precision: fraction of returned docs containing the ref
            def precision_hits(docs):
                if len(docs) == 0:
                    return 0.0
                hits = 0
                for d in docs:
                    if d and ref.lower() in d.lower():
                        hits += 1
                return hits / len(docs)

            rows.append({
                'query': q,
                'reference': ref,
                'baseline_hit': base_hit,
                'rerank_hit': rr_hit,
                'baseline_precision': precision_hits(base_docs),
                'rerank_precision': precision_hits(rr_docs)
            })

            progress.progress(int((i+1)/total*100))

        res_df = _pd.DataFrame(rows)
        # aggregated
        baseline_recall = res_df['baseline_hit'].mean()
        rerank_recall = res_df['rerank_hit'].mean()
        baseline_prec = res_df['baseline_precision'].mean()
        rerank_prec = res_df['rerank_precision'].mean()

        st.sidebar.success("Benchmark complete")
        st.subheader("Benchmark results")
        st.write(f"Baseline recall@{final_k}: {baseline_recall:.3f}")
        st.write(f"Rerank recall@{final_k}: {rerank_recall:.3f}")
        st.write(f"Baseline precision@{final_k}: {baseline_prec:.3f}")
        st.write(f"Rerank precision@{final_k}: {rerank_prec:.3f}")

        st.download_button("Download per-query results (CSV)", res_df.to_csv(index=False), file_name='benchmark_results.csv')