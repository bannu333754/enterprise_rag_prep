import os
import shutil
import chromadb
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 13 Hybrid Search RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day13_hybrid_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day13"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory document storage for BM25
DOC_STORE = []  # List of dicts: {"id": str, "text": str, "source": str, "page": int}
bm25_index = None

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class HybridRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    retrieval_type: str
    text: str

class HybridResponse(BaseModel):
    session_id: str
    standalone_question: str
    sources: list[SourceDetail]
    answer: str

def rebuild_bm25():
    global bm25_index
    if not DOC_STORE:
        bm25_index = None
        return
    tokenized_corpus = [doc["text"].lower().split() for doc in DOC_STORE]
    bm25_index = BM25Okapi(tokenized_corpus)

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 13 Hybrid Search (BM25 + Dense + RRF)",
        "indexed_chunks": len(DOC_STORE),
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

            DOC_STORE.append({
                "id": doc_id,
                "text": chunk,
                "source": file.filename,
                "page": page_idx + 1
            })
            total_added += 1

    rebuild_bm25()

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_collection_chunks": len(DOC_STORE)
    }

def rewrite_query(history: List[Dict[str, str]], question: str) -> str:
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"""Rewrite the follow-up question into a complete search query.
Return ONLY the query.

History:
{history_str}

Follow-up: {question}
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

@app.post("/chat", response_model=HybridResponse)
def hybrid_chat(payload: HybridRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not DOC_STORE or bm25_index is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    standalone_query = rewrite_query(history, payload.question)

    # 1. Dense Vector Search
    query_vector = embed_model.encode(standalone_query).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None

    n_candidates = min(4, len(DOC_STORE))
    dense_res = collection.query(
        query_embeddings=[query_vector],
        n_results=n_candidates,
        where=where_filter
    )
    dense_ids = dense_res["ids"][0] if dense_res["ids"] else []

    # 2. Sparse BM25 Search
    tokenized_query = standalone_query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # Filter by target_doc if specified
    filtered_indices = [
        i for i, doc in enumerate(DOC_STORE)
        if payload.target_doc == "ALL" or doc["source"] == payload.target_doc
    ]
    bm25_ranked_indices = sorted(filtered_indices, key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
    bm25_ids = [DOC_STORE[i]["id"] for i in bm25_ranked_indices]

    # 3. Reciprocal Rank Fusion (RRF) with constant k=60
    rrf_scores: Dict[str, float] = {}
    k = 60

    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort candidates by combined RRF score
    sorted_candidate_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)[:4]

    doc_lookup = {d["id"]: d for d in DOC_STORE}
    candidate_items = [doc_lookup[did] for did in sorted_candidate_ids if did in doc_lookup]

    # 4. Cross-Encoder Re-Ranking on Fusion Candidates
    pair_inputs = [[standalone_query, item["text"]] for item in candidate_items]
    scores = reranker.predict(pair_inputs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    selected_sources = []
    combined_context = []

    for rank, idx in enumerate(ranked_indices):
        item = candidate_items[idx]
        score = float(scores[idx])

        # Check retrieval origin
        is_dense = item["id"] in dense_ids
        is_bm25 = item["id"] in bm25_ids
        origin = "Hybrid (Both)" if (is_dense and is_bm25) else ("Dense Vector" if is_dense else "BM25 Keyword")

        selected_sources.append(SourceDetail(
            doc=item["source"],
            page=item["page"],
            score=round(score, 3),
            retrieval_type=origin,
            text=item["text"]
        ))
        combined_context.append(f"[Source {rank+1}: {item['source']} (Page {item['page']})]\n{item['text']}")

    context_str = "\n\n".join(combined_context)

    system_prompt = (
        "You are an enterprise research assistant using hybrid retrieval. "
        "Answer accurately based on the context blocks and chat history. Quote exact names, IDs, or values if present."
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
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return HybridResponse(
        session_id=session_id,
        standalone_question=standalone_query,
        sources=selected_sources,
        answer=answer
    )
