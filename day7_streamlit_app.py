import streamlit as st
import requests

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖")

st.title("🤖 Enterprise Multi-Doc RAG Platform")
st.caption("Ask questions against your indexed PDF knowledge base in real-time.")

FASTAPI_URL = "http://127.0.0.1:8000/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message and message["source"]:
            with st.expander("📄 View Retrieved Source Chunk"):
                st.write(message["source"])

user_prompt = st.chat_input("Ask a question about the document...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving facts & querying Gemini..."):
            try:
                response = requests.post(
                    FASTAPI_URL,
                    json={"question": user_prompt},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    source_chunk = data.get("retrieved_chunk", "No source context available.")

                    st.markdown(answer)
                    with st.expander("📄 View Retrieved Source Chunk"):
                        st.write(source_chunk)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source": source_chunk
                    })
                else:
                    st.error(f"Backend Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend! Ensure `python -m uvicorn day6_rag_api:app --host 127.0.0.1 --port 8000` is running.")