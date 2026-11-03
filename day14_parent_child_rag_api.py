import os
import shutil
import chromadb
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 14 Parent-Child Hierarchical RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
child_collection = chroma_client.get_or_create_collection(name="day14_child_chunks")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day14"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Parent Document Store
PARENT_STORE: Dict[str, Dict] = {}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class ParentChildRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    matched_child: str
    parent_context: str

class ParentChildResponse(BaseModel):
    session_id: str
    standalone_question: str
    sources: list[SourceDetail]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 14 Hierarchical Parent-Child RAG",
        "parent_blocks": len(PARENT_STORE),
        "indexed_child_chunks": child_collection.count(),
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
    total_parents = 0
    total_children = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        parent_size = 180
        child_size = 45

        # 1. Break into Parent Chunks
        parent_splits = [" ".join(words[i:i + parent_size]) for i in range(0, len(words), parent_size)]

        for p_idx, parent_text in enumerate(parent_splits):
            parent_id = f"{file.filename}_p{page_idx + 1}_parent_{p_idx}"
            PARENT_STORE[parent_id] = {
                "source": file.filename,
                "page": page_idx + 1,
                "text": parent_text
            }
            total_parents += 1

            # 2. Break Parent into Child Chunks
            p_words = parent_text.split()
            child_splits = [" ".join(p_words[j:j + child_size]) for j in range(0, len(p_words), child_size)]

            for c_idx, child_text in enumerate(child_splits):
                child_id = f"{parent_id}_child_{c_idx}"
                vector = embed_model.encode(child_text).tolist()

                child_collection.add(
                    documents=[child_text],
                    embeddings=[vector],
                    metadatas=[{
                        "parent_id": parent_id,
                        "source": file.filename,
                        "page": page_idx + 1
                    }],
                    ids=[child_id]
                )
                total_children += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "parent_chunks_created": total_parents,
        "child_chunks_indexed": total_children,
        "total_children_in_db": child_collection.count()
    }

def rewrite_query(history: List[Dict[str, str]], question: str) -> str:
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"""Rewrite follow-up question to be self-contained for document search.
Return ONLY the rewritten query text.

Conversation History:
{history_str}

Question: {question}
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

@app.post("/chat", response_model=ParentChildResponse)
def hierarchical_chat(payload: ParentChildRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if child_collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    standalone_query = rewrite_query(history, payload.question)

    # 1. Search across precise Child Chunks
    query_vector = embed_model.encode(standalone_query).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None

    n_candidates = min(6, child_collection.count())
    results = child_collection.query(
        query_embeddings=[query_vector],
        n_results=n_candidates,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching content found.")

    child_docs = results['documents'][0]
    metas = results['metadatas'][0]

    # 2. Cross-Encoder Re-Ranking on Child matches
    pair_inputs = [[standalone_query, c_text] for c_text in child_docs]
    scores = reranker.predict(pair_inputs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # 3. Resolve to Unique Parent Chunks (Deduplication)
    selected_sources = []
    combined_context = []
    seen_parents = set()

    for idx in ranked_indices:
        parent_id = metas[idx]["parent_id"]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        parent_info = PARENT_STORE.get(parent_id, {})
        parent_text = parent_info.get("text", child_docs[idx])
        score = float(scores[idx])
        source_name = metas[idx].get("source", "Unknown")
        page_num = metas[idx].get("page", 0)

        selected_sources.append(SourceDetail(
            doc=source_name,
            page=page_num,
            score=round(score, 3),
            matched_child=child_docs[idx],
            parent_context=parent_text
        ))

        combined_context.append(f"[Source: {source_name} (Page {page_num})]\n{parent_text}")

        if len(selected_sources) == 2:
            break

    context_str = "\n\n".join(combined_context)

    # 4. LLM Synthesis with Rich Parent Context
    system_prompt = (
        "You are an enterprise research specialist. You have been provided with comprehensive parent context blocks "
        "retrieved via hierarchical child-matching. Answer the query thoroughly and accurately based on the context."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- PARENT CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    })

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return ParentChildResponse(
        session_id=session_id,
        standalone_question=standalone_query,
        sources=selected_sources,
        answer=answer
    )
