import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 9 Advanced RAG with Re-ranking", layout="wide")

st.title("⚡ Day 9: RAG with Cross-Encoder Re-ranking")
st.caption("Dense vector retrieval (Top-4) + Cross-Encoder re-ranking (Top-2) + Groq Synthesis")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("📁 Document Manager")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
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
            st.success(f"Successfully indexed! Total new chunks: {total_added}")
        else:
            st.warning("Upload a PDF file first.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Scope", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question across your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {"question": user_query, "target_doc": selected_doc}

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, re-ranking, and generating answer..."):
            try:
                res = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    # Expandable section to show re-ranked context & relevance scores
                    with st.expander("🔍 View Re-ranked Sources & Re-rank Scores"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Rank {i+1}** | **Doc:** `{src['doc']}` (Page {src['page']}) | **Re-rank Score:** `{src['score']}`")
                            st.caption(f"\"{src['text']}\"")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    st.error(f"Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")