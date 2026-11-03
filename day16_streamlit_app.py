import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 16 GraphRAG Platform", layout="wide")

st.title("🕸️ Day 16: GraphRAG (Knowledge Graph + Vector Search)")
st.caption("Multi-hop Entity-Relation Reasoning + ChromaDB Dense Vector Fusion")

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
    
    if st.button("Index Documents & Build KG"):
        if uploaded_files:
            total_added = 0
            total_triplets = 0
            for file in uploaded_files:
                files = {"file": (file.name, file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        total_added += data.get("chunks_indexed", 0)
                        total_triplets += data.get("graph_triplets_created", 0)
                        if file.name not in st.session_state.uploaded_files_list:
                            st.session_state.uploaded_files_list.append(file.name)
                    else:
                        st.error(f"Error on {file.name}: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            st.success(f"Built Knowledge Graph! Indexed {total_added} chunks & extracted {total_triplets} relations.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to see Graph Traversal + Vector synthesis...")

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
        with st.spinner("Traversing Knowledge Graph entities & re-ranking text chunks..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    entities = data.get("extracted_entities", [])
                    triplets = data.get("graph_triplets", [])
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    with st.expander("🕸️ GraphRAG Telemetry & Entity Links"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Matched Entities in Graph:**")
                            if entities:
                                for ent in entities:
                                    st.write(f"- `{ent.title()}`")
                            else:
                                st.caption("No direct entity match in query.")
                        with col2:
                            st.markdown("**Traversed Relationships:**")
                            if triplets:
                                for trip in triplets:
                                    st.code(trip, language="markdown")
                            else:
                                st.caption("No graph paths traversed.")

                        st.divider()
                        st.markdown("**Dense Vector Retrievial Chunks:**")
                        for i, src in enumerate(sources):
                            st.markdown(f"**Rank {i+1}** | **Doc:** `{src['doc']}` (p.{src['page']}) | **Score:** `{src['score']}`")
                            st.caption(f"\"{src['text']}\"")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
