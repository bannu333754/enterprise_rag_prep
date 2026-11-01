import streamlit as st
import requests

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Enterprise Multi-Doc RAG Platform")
st.caption("Ask questions against your indexed PDF knowledge base in real-time.")

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/query"

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "source" in msg:
            with st.expander("📄 View Retrieved Source Chunk"):
                st.write(msg["source"])

# Accept user input
if user_prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving facts & generating response..."):
            try:
                response = requests.post(API_URL, json={"question": user_prompt}, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    source_chunk = data.get("retrieved_chunk", "No source context available.")

                    st.markdown(answer)
                    with st.expander("📄 View Retrieved Source Chunk"):
                        st.write(source_chunk)

                    # Save assistant reply to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source": source_chunk
                    })
                else:
                    st.error(f"Backend Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend! Ensure `uvicorn day6_rag_api:app --reload` is running on port 8000.")