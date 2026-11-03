import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 19 Semantic Caching RAG", layout="wide")

st.title("⚡ Day 19: Semantic Caching & Vector Lookup")
st.caption("Sub-millisecond Approximate Query Caching with Cosine Similarity Thresholds")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Session & Cache")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")
    
    use_cache_toggle = st.toggle("Enable Semantic Cache", value=True)

    if st.button("🗑️ Clear Cache Memory"):
        try:
            res = requests.post(f"{API_URL}/clear_cache")
            if res.status_code == 200:
                st.success("Semantic Cache Cleared!")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Index Documents"):
        if uploaded_files:
            total_added = 0
            for file in uploaded_files:
                files = {"file": (file.name, file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        total_added += data.get("chunks_indexed", 0)
                        if file.name not in st.session_state.uploaded_files_list:
                            st.session_state.uploaded_files_list.append(file.name)
                    else:
                        st.error(f"Error on {file.name}: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            st.success(f"Indexed successfully! Added {total_added} chunks.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question (or a paraphrased version to test cache hit)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "use_cache": use_cache_toggle
    }

    with st.chat_message("assistant"):
        with st.spinner("Checking semantic vector cache & processing query..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    status = data.get("cache_status", "MISS")
                    lat = data.get("latency_ms", 0.0)
                    sim = data.get("similarity_score", 0.0)
                    matched_q = data.get("cache_matched_query")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    # Performance Dashboard
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if status == "HIT":
                            st.success(f"⚡ Cache Status: **{status}**")
                        else:
                            st.info(f"🔄 Cache Status: **{status}**")
                    with col2:
                        st.metric("Total Latency", f"{lat:.1f} ms")
                    with col3:
                        st.metric("Similarity Score", f"{sim:.4f}")

                    with st.expander("🔬 Semantic Cache Provenance"):
                        if status == "HIT":
                            st.markdown(f"**Matched Historical Query:** `{matched_q}`")
                            st.caption("Returned instant response directly from in-memory semantic cache. Zero Groq tokens consumed.")
                        else:
                            st.caption("Query vector did not exceed semantic similarity threshold (>= 0.90). Full retrieval & LLM generation triggered.")

                        st.divider()
                        st.markdown("**Associated Grounded Sources:**")
                        for i, src in enumerate(sources):
                            st.markdown(f"**Rank {i+1}** | **Doc:** `{src['doc']}` (p.{src['page']}) | **Score:** `{src['score']}`")
                            st.caption(f"\"{src['text']}\"")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
