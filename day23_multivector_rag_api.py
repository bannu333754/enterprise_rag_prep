import os
import uuid
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 23 Multi-Vector Summary RAG API", version="1.0.0")

# Embedding & CrossEncoder Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day23_multivector_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day23"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-Memory Document Store: maps chunk_id -> raw full-size text chunk
RAW_DOC_STORE: Dict[str, Dict] = {}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class MultiVectorRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class RetrievedResult(BaseModel):
    chunk_id: str
    source: str
    page: int
    matched_summary: str
    raw_content: str
    score: float

class MultiVectorResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_items: List[RetrievedResult]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 23 Multi-Vector Summary RAG",
        "summary_vectors_indexed": collection.count(),
        "raw_chunks_stored": len(RAW_DOC_STORE),
        "active_sessions": len(SESSION_HISTORIES)
    }

def generate_chunk_summary(chunk_text: str) -> str:
    """Uses Groq LLM to generate a dense, high-signal proposition summary for indexing."""
    prompt = f"""Summarize the following passage into a single, dense semantic proposition sentence capturing its core facts and technical meaning:

Passage:
{chunk_text}

Dense Semantic Summary:"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content.strip()
    except Exception:
        # Fallback to extractive first sentences if LLM call fails
        words = chunk_text.split()
        return " ".join(words[:25]) + "..."

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
        chunk_size = 120  # Larger full-context raw chunks
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, raw_chunk in enumerate(chunks):
            chunk_id = f"{file.filename}_p{page_idx + 1}_c{c_idx}_{str(uuid.uuid4())[:4]}"
            
            # 1. Store full raw text chunk in Key-Value store
            RAW_DOC_STORE[chunk_id] = {
                "source": file.filename,
                "page": page_idx + 1,
                "raw_text": raw_chunk
            }

            # 2. Generate proposition summary for vector embedding
            summary = generate_chunk_summary(raw_chunk)
            summary_vector = embed_model.encode(summary).tolist()

            # 3. Add to ChromaDB: Vector is created on SUMMARY, metadata points to chunk_id
            collection.add(
                documents=[summary],
                embeddings=[summary_vector],
                metadatas=[{
                    "source": file.filename,
                    "page": page_idx + 1,
                    "chunk_id": chunk_id
                }],
                ids=[f"vec_{chunk_id}"]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "multi_vectors_indexed": total_added,
        "total_summaries": collection.count()
    }

@app.post("/chat", response_model=MultiVectorResponse)
def multi_vector_chat(payload: MultiVectorRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Semantic search against dense proposition summaries
    query_vector = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching document summaries found.")

    matched_summaries = results['documents'][0]
    matched_metas = results['metadatas'][0]

    # Step 2: Retrieve full raw chunks from RAW_DOC_STORE
    raw_candidates = []
    for meta in matched_metas:
        c_id = meta.get("chunk_id")
        stored_item = RAW_DOC_STORE.get(c_id, {})
        raw_candidates.append(stored_item.get("raw_text", ""))

    # Step 3: Cross-Encoder re-rank using full raw content against query
    pairs = [[payload.question, doc] for doc in raw_candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    retrieved_items: List[RetrievedResult] = []
    context_blocks = []

    for rank, idx in enumerate(ranked_indices):
        meta = matched_metas[idx]
        c_id = meta.get("chunk_id", "unknown")
        raw_text = raw_candidates[idx]
        summary_text = matched_summaries[idx]
        score = float(scores[idx])

        retrieved_items.append(RetrievedResult(
            chunk_id=c_id,
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            matched_summary=summary_text,
            raw_content=raw_text,
            score=round(score, 3)
        ))
        context_blocks.append(f"\n{raw_text}")

    context_str = "\n\n".join(context_blocks)

    # Step 4: Final LLM synthesis
    system_prompt = (
        "You are an enterprise research assistant. Answer the user query comprehensively and "
        "factually using only the grounded raw context retrieved via Multi-Vector Indexing."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    })

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        final_answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return MultiVectorResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_items=retrieved_items,
        final_answer=final_answer
    )
