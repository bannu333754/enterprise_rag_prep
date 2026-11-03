import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 24 Adaptive Complexity RAG", layout="wide")

st.title("🎛️ Day 24: Adaptive RAG & Dynamic Strategy Selection")
st.caption("Complexity-Driven Routing (No-Retrieval / Single-Pass / Multi-Hop Iterative Decomposition)")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Adaptive Config")
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

user_query = st.chat_input("Test a greeting, factual query, or complex comparison...")

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
        with st.spinner("Analyzing query complexity & dispatching routing strategy..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    decision = data.get("decision", {})
                    sources = data.get("retrieved_sources", [])

                    st.markdown(answer)

                    # Routing Strategy Badge
                    tier = decision.get("tier", "SINGLE_PASS")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if tier == "NO_RETRIEVAL":
                            st.info("⚡ Tier: **NO_RETRIEVAL** (Direct LLM)")
                        elif tier == "SINGLE_PASS":
                            st.success("🎯 Tier: **SINGLE_PASS** (Standard RAG)")
                        else:
                            st.warning("🔄 Tier: **MULTI_HOP** (Decomposition RAG)")
                    with col2:
                        st.caption(f"**Router Reason:** {decision.get('reasoning')}")

                    if decision.get("sub_queries") and tier == "MULTI_HOP":
                        st.markdown(f"**Decomposed Sub-Queries:** `{decision.get('sub_queries')}`")

                    if sources:
                        with st.expander(f"📚 Retrieved Context Sources ({len(sources)} passages)"):
                            for i, s in enumerate(sources):
                                st.markdown(f"**Rank {i+1}** | `{s['source']}` (p.{s['page']}) | Re-rank Score: `{s['score']}`")
                                st.caption(f"\"{s['text']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
