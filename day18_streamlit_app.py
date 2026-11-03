import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 18 Contextual Compression RAG", layout="wide")

st.title("🗜️ Day 18: Contextual Compression & Extractive Pruning")
st.caption("Sentence-Level Cross-Encoder Scoring to Suppress Context Noise and Minimize Latency")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Session & Docs")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")
    
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    enable_comp = st.toggle("Enable Contextual Compression", value=True)
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

user_query = st.chat_input("Ask a question to test sentence pruning & context savings...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "enable_compression": enable_comp
    }

    with st.chat_message("assistant"):
        with st.spinner("Pruning noise sentences & re-compressing context..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    raw_w = data.get("total_raw_words", 0)
                    comp_w = data.get("total_compressed_words", 0)
                    savings = data.get("net_savings_pct", 0.0)
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    # Observability Metrics Cards
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Raw Retrieved Words", f"{raw_w} words")
                    with col2:
                        st.metric("Compressed Context", f"{comp_w} words")
                    with col3:
                        st.metric("Token/Noise Pruned", f"{savings}% saved", delta=f"-{raw_w - comp_w} words")

                    with st.expander("🔬 Sentence-Level Extractive Pruning Details"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}:** `{src['doc']}` (Page {src['page']}) | **Pruned:** `{src['compression_ratio_pct']}%`")
                            st.markdown("**Retained Sentences (High-Relevance):**")
                            for s in src["extracted_sentences"]:
                                st.write(f"• *\"{s}\"*")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
