import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 22 Iterative Self-Corrective RAG", layout="wide")

st.title("🔁 Day 22: Self-Corrective Iterative RAG")
st.caption("Automated Gap Analysis, Confidence Auditing & Iterative Re-Retrieval Loops")

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

user_query = st.chat_input("Ask a question to test iterative gap-filling and re-retrieval...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc
    }

    with st.chat_message("assistant"):
        with st.spinner("Iterative RAG evaluating answer completeness & gaps..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    iters = data.get("total_iterations", 1)
                    history_traces = data.get("iteration_history", [])

                    st.markdown(answer)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Iterative Loops", f"{iters} passes")
                    with col2:
                        last_conf = history_traces[-1]["evaluation_confidence"] if history_traces else 1.0
                        st.metric("Final Confidence", f"{last_conf}")

                    if history_traces:
                        with st.expander("🔬 Iterative Self-Correction Trace"):
                            for tr in history_traces:
                                st.markdown(f"**Iteration {tr['iteration_number']}:** Query: `{tr['query_used']}` ({tr['retrieved_chunks_count']} chunks)")
                                st.caption(f"Confidence: {tr['evaluation_confidence']} | Gap Analysis: {tr['missing_info_gaps']}")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
