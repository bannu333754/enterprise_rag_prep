import os
import shutil
import chromadb
from typing import Dict, List, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 11 Agentic RAG with Dynamic Router", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day11_agentic_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day11"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class AgentRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class AgentResponse(BaseModel):
    session_id: str
    route_decision: str
    standalone_question: str
    sources: list[SourceDetail]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 11 Agentic Router RAG",
        "indexed_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    total_added = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 60
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{"source": file.filename, "page": page_idx + 1}],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_collection_chunks": collection.count()
    }

def route_query(question: str) -> str:
    """Decides whether the query needs document search or general reasoning."""
    prompt = f"""You are a routing supervisor in an enterprise AI system.
Classify the user's intent into EXACTLY ONE category:
- DOCUMENT_SEARCH : If the question asks about specific uploaded files, certificates, reports, names, ids, policies, or specific facts likely contained in an uploaded document.
- GENERAL_CONVERSATION : If the question is a greeting, general chat, math calculation, general coding question, or world knowledge not specific to user's private documents.

Respond ONLY with the category name: DOCUMENT_SEARCH or GENERAL_CONVERSATION.

User Query: {question}
Category:"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        decision = completion.choices[0].message.content.strip()
        if "DOCUMENT_SEARCH" in decision:
            return "DOCUMENT_SEARCH"
        return "GENERAL_CONVERSATION"
    except Exception:
        return "DOCUMENT_SEARCH"

def rewrite_query(history: List[Dict[str, str]], question: str) -> str:
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question to be a complete self-contained query.
Return ONLY the rewritten query text.

Conversation History:
{history_str}

Follow-up Question: {question}
Rewritten Query:"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content.strip() or question
    except Exception:
        return question

@app.post("/agent_chat", response_model=AgentResponse)
def agent_chat(payload: AgentRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # 1. Routing Step
    route = route_query(payload.question)

    # Branch A: General Conversation Route (No DB retrieval needed)
    if route == "GENERAL_CONVERSATION":
        messages = [
            {"role": "system", "content": "You are a helpful, professional AI assistant. Respond warmly and directly."}
        ]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": payload.question})

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.3
        )
        answer = completion.choices[0].message.content

        history.append({"role": "user", "content": payload.question})
        history.append({"role": "assistant", "content": answer})

        return AgentResponse(
            session_id=session_id,
            route_decision="GENERAL_CONVERSATION (Bypassed Vector DB)",
            standalone_question=payload.question,
            sources=[],
            answer=answer
        )

    # Branch B: Document Search Route
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    standalone_query = rewrite_query(history, payload.question)
    query_vector = embed_model.encode(standalone_query).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None

    n_candidates = min(4, collection.count())
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_candidates,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Cross-Encoder Re-ranking
    pair_inputs = [[standalone_query, doc_text] for doc_text in candidates]
    scores = reranker.predict(pair_inputs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    selected_sources = []
    combined_context = []

    for rank, idx in enumerate(ranked_indices):
        chunk_text = candidates[idx]
        meta = metas[idx]
        score = float(scores[idx])

        selected_sources.append(SourceDetail(
            doc=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            score=round(score, 3),
            text=chunk_text
        ))
        combined_context.append(f"[Source {rank+1}: {meta.get('source')} (Page {meta.get('page')})]\n{chunk_text}")

    context_str = "\n\n".join(combined_context)

    system_prompt = (
        "You are an enterprise document specialist. Answer accurately using the retrieved context blocks and chat history. "
        "Cite the document name and page number when stating specific facts."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    })

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.2
    )
    answer = completion.choices[0].message.content

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return AgentResponse(
        session_id=session_id,
        route_decision="DOCUMENT_SEARCH (ChromaDB + CrossEncoder)",
        standalone_question=standalone_query,
        sources=selected_sources,
        answer=answer
    )
