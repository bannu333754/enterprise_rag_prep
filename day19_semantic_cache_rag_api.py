import os
import time
import shutil
import chromadb
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 19 Semantic Caching RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day19_semantic_cache_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day19"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Semantic Cache: List of dicts -> {"query": str, "vector": np.ndarray, "answer": str, "sources": list}
SEMANTIC_CACHE: List[Dict] = []
CACHE_SIMILARITY_THRESHOLD = 0.90
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class CacheRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    use_cache: bool = True

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class CacheResponse(BaseModel):
    session_id: str
    standalone_question: str
    cache_status: str  # "HIT" or "MISS"
    cache_matched_query: Optional[str] = None
    similarity_score: float
    latency_ms: float
    sources: list[SourceDetail]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 19 Semantic Caching RAG",
        "cache_entries": len(SEMANTIC_CACHE),
        "total_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/clear_cache")
def clear_cache():
    global SEMANTIC_CACHE
    SEMANTIC_CACHE.clear()
    return {"status": "Cache Cleared", "entries": len(SEMANTIC_CACHE)}

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
        chunk_size = 70
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

def rewrite_query(history: List[Dict[str, str]], question: str) -> str:
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"""Rewrite the follow-up question to be self-contained for search. Return ONLY the rewritten text.
History:
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

def check_semantic_cache(query_vector: np.ndarray) -> tuple[Optional[Dict], float]:
    """Finds the closest query in cache using cosine similarity."""
    if not SEMANTIC_CACHE:
        return None, 0.0

    best_match = None
    best_similarity = -1.0

    # Normalize query vector for cosine similarity
    norm_q = query_vector / np.linalg.norm(query_vector)

    for entry in SEMANTIC_CACHE:
        cached_vec = entry["vector"]
        norm_c = cached_vec / np.linalg.norm(cached_vec)
        sim = float(np.dot(norm_q, norm_c))

        if sim > best_similarity:
            best_similarity = sim
            best_match = entry

    if best_similarity >= CACHE_SIMILARITY_THRESHOLD:
        return best_match, best_similarity
    return None, best_similarity

@app.post("/chat", response_model=CacheResponse)
def semantic_cache_chat(payload: CacheRequest):
    start_time = time.time()

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    standalone_query = rewrite_query(history, payload.question)
    query_vector_raw = embed_model.encode(standalone_query)
    query_vector = np.array(query_vector_raw, dtype=np.float32)

    # 1. Evaluate Semantic Cache Hit
    if payload.use_cache:
        cached_entry, sim_score = check_semantic_cache(query_vector)
        if cached_entry:
            latency = (time.time() - start_time) * 1000.0
            history.append({"role": "user", "content": payload.question})
            history.append({"role": "assistant", "content": cached_entry["answer"]})
            return CacheResponse(
                session_id=session_id,
                standalone_question=standalone_query,
                cache_status="HIT",
                cache_matched_query=cached_entry["query"],
                similarity_score=round(sim_score, 4),
                latency_ms=round(latency, 2),
                sources=cached_entry["sources"],
                answer=cached_entry["answer"]
            )
    else:
        sim_score = 0.0

    # 2. Cache Miss: Execute ChromaDB Retrieval & Re-ranking
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_candidates = min(4, collection.count())
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_candidates,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pair_inputs = [[standalone_query, doc] for doc in candidates]
    scores = reranker.predict(pair_inputs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    selected_sources = []
    combined_context = []

    for rank, idx in enumerate(ranked_indices):
        chunk_text = candidates[idx]
        meta = metas[idx]
        score = float(scores[idx])

        src = SourceDetail(
            doc=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            score=round(score, 3),
            text=chunk_text
        )
        selected_sources.append(src)
        combined_context.append(f"[Source {rank+1}: {meta.get('source')} (Page {meta.get('page')})]\n{chunk_text}")

    context_str = "\n\n".join(combined_context)

    # 3. LLM Synthesis
    system_prompt = "You are an enterprise research assistant. Answer accurately based on the context."
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
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Save to Semantic Cache
    SEMANTIC_CACHE.append({
        "query": standalone_query,
        "vector": query_vector,
        "answer": answer,
        "sources": selected_sources
    })

    latency = (time.time() - start_time) * 1000.0
    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return CacheResponse(
        session_id=session_id,
        standalone_question=standalone_query,
        cache_status="MISS",
        cache_matched_query=None,
        similarity_score=round(sim_score, 4),
        latency_ms=round(latency, 2),
        sources=selected_sources,
        answer=answer
    )
