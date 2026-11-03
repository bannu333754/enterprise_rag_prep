import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 11 Agentic Router RAG", layout="wide")

st.title("🤖 Day 11: Agentic RAG with Dynamic Router")
st.caption("Intelligent Query Routing + Vector Database Bypass + Cross-Encoder Re-Ranking")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Agent Controls")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")
    
    if st.button("🔄 Reset Session"):
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
    selected_doc = st.selectbox("Filter Document Scope", options=doc_options)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask anything (document question or general query)...")

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
        with st.spinner("Agent evaluating routing path & synthesizing answer..."):
            try:
                res = requests.post(f"{API_URL}/agent_chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    route = data.get("route_decision", "")
                    standalone_q = data.get("standalone_question", "")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    with st.expander(f"🚦 Agent Routing Telemetry: `{route}`"):
                        st.write(f"**Reformulated Search Query:** `{standalone_q}`")
                        if sources:
                            st.write("**Retrieved & Re-ranked Sources:**")
                            for i, src in enumerate(sources):
                                st.markdown(f"**Rank {i+1}** | **Doc:** `{src['doc']}` (Page {src['page']}) | **Score:** `{src['score']}`")
                                st.caption(f"\"{src['text']}\"")
                        else:
                            st.info("Direct synthesis applied: No vector database search needed for this prompt.")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
