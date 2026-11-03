import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 20 Guardrails & Self-Reflective RAG", layout="wide")

st.title("🛡️ Day 20: Self-Reflective Guardrails & Hallucination Defense")
st.caption("Prompt Injection Screening + PII Redaction + Self-Reflective Groundedness Auditing")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Safety & Config")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")
    
    guardrails_toggle = st.toggle("Enable Enterprise Guardrails", value=True)

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

user_query = st.chat_input("Ask a question, test prompt injection, or input sensitive PII...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "enable_guardrails": guardrails_toggle
    }

    with st.chat_message("assistant"):
        with st.spinner("Screening guardrails & conducting self-reflection audit..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    audit = data.get("safety_audit", {})
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if audit.get("injection_detected"):
                            st.error("🚨 Injection: Detected")
                        else:
                            st.success("🛡️ Input Security: Clean")
                    with col2:
                        st.metric("PII Redacted", f"{audit.get('pii_redacted_count', 0)} entities")
                    with col3:
                        conf = audit.get("groundedness_confidence", 1.0)
                        if audit.get("hallucination_detected"):
                            st.error(f"⚠️ Grounding: {conf:.2f}")
                        else:
                            st.metric("Grounding Confidence", f"{conf:.2f} / 1.00")

                    with st.expander("🔬 Guardrail Audit Inspection Details"):
                        st.markdown(f"**Sanitized Query:** `{data.get('standalone_question')}`")
                        st.markdown(f"**Self-Reflection Audit Notes:** {audit.get('audit_notes')}")
                        st.divider()
                        st.markdown("**Grounded Sources Retrieved:**")
                        for i, src in enumerate(sources):
                            st.markdown(f"**Rank {i+1}** | **Doc:** `{src['doc']}` (p.{src['page']}) | **Score:** `{src['score']}`")
                            st.caption(f"\"{src['text']}\"")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
