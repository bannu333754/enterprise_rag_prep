import streamlit as st
import requests

st.set_page_config(page_title="Multi-Doc Enterprise RAG", page_icon="📚", layout="wide")

st.title("📚 Dynamic Multi-Document Enterprise RAG")
st.caption("Upload multiple PDFs dynamically, index with metadata, and query selectively.")

FASTAPI_BASE = "http://127.0.0.1:8000"

with st.sidebar:
    st.header("📂 Document Manager")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Index Documents") and uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Ingesting {file.name}..."):
                files = {"file": (file.name, file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{FASTAPI_BASE}/upload", files=files, timeout=60)
                    if res.status_code == 200:
                        st.success(f"Indexed: {file.name}")
                    else:
                        st.error(f"Failed {file.name}: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    st.divider()
    filter_choice = st.selectbox(
        "Filter Search Domain",
        options=["ALL"] + [f.name for f in uploaded_files] if uploaded_files else ["ALL"]
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            with st.expander(f"📄 Source: {msg['meta']['source']} (Page {msg['meta']['page']})"):
                st.write(msg["meta"]["chunk"])

user_query = st.chat_input("Ask a question across indexed documents...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Executing semantic retrieval & synthesis..."):
            try:
                res = requests.post(
                    f"{FASTAPI_BASE}/query",
                    json={"question": user_query, "target_doc": filter_choice},
                    timeout=60
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    meta_info = {
                        "source": data["source_doc"],
                        "page": data["page_num"],
                        "chunk": data["retrieved_chunk"]
                    }

                    st.markdown(answer)
                    with st.expander(f"📄 Source: {meta_info['source']} (Page {meta_info['page']})"):
                        st.write(meta_info["chunk"])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": meta_info
                    })
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("FastAPI backend is offline. Run `day8_rag_api.py` first.")