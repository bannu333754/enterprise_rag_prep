import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 17 HyDE & Multi-Query Fusion RAG", layout="wide")

st.title("💡 Day 17: HyDE & Multi-Query Fusion RAG")
st.caption("Hypothetical Document Embeddings + Multi-Perspective Query Decomposition + Cross-Encoder Fusion")

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
    enable_hyde = st.toggle("Enable HyDE Generator", value=True)
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

user_query = st.chat_input("Ask a question to test HyDE and query decomposition...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "use_hyde": enable_hyde
    }

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing hypothetical document & fusing query variations..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    hypo_doc = data.get("generated_hypothetical_doc", "")
                    queries = data.get("expanded_queries", [])
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    with st.expander("🔬 HyDE & Query Decomposition Telemetry"):
                        st.markdown("**🧠 Generated Hypothetical Document (Doc-to-Doc Embedding):**")
                        st.info(hypo_doc)

                        st.markdown("**🔀 Multi-Query Variations:**")
                        for q in queries:
                            st.write(f"- `{q}`")

                        st.divider()
                        st.markdown("**🎯 Re-ranked Top Chunks:**")
                        for i, src in enumerate(sources):
                            st.markdown(f"**Rank {i+1}** | **Doc:** `{src['doc']}` (p.{src['page']}) | **Cross-Encoder Score:** `{src['score']}`")
                            st.caption(f"\"{src['text']}\"")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
