import os
import json
import shutil
import chromadb
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 17 HyDE & Multi-Query Fusion RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day17_hyde_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day17"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class HyDERequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    use_hyde: bool = True

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class HyDEResponse(BaseModel):
    session_id: str
    original_question: str
    generated_hypothetical_doc: str
    expanded_queries: list[str]
    sources: list[SourceDetail]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 17 HyDE + Multi-Query RAG",
        "total_chunks": collection.count(),
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
        chunk_size = 65
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

def generate_hyde_doc(question: str) -> str:
    """Generates a hypothetical passage that answers the question."""
    prompt = f"""Write a concise, professional hypothetical paragraph that answers this question. 
It does not need to be factual, but should mirror the scientific, corporate, or technical vocabulary expected in an official document.

Question: {question}
Hypothetical Passage:"""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return question

def generate_multi_queries(question: str) -> List[str]:
    """Generates 3 different variations/perspectives of the user question."""
    prompt = f"""Generate 3 distinct search variations/perspectives for the user question.
Return STRICTLY a JSON array of 3 strings.

Question: {question}
"""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = completion.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception:
        return [question]

@app.post("/chat", response_model=HyDEResponse)
def hyde_chat(payload: HyDERequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # 1. Generate Hypothetical Document & Query Variations
    hypo_doc = generate_hyde_doc(payload.question) if payload.use_hyde else payload.question
    variations = generate_multi_queries(payload.question)
    search_queries = list(set([payload.question, hypo_doc] + variations))

    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None

    # 2. Multi-Vector Candidate Retrieval
    all_candidate_chunks = {}
    for sq in search_queries:
        vector = embed_model.encode(sq).tolist()
        n_res = min(3, collection.count())
        res = collection.query(query_embeddings=[vector], n_results=n_res, where=where_filter)
        if res['documents'] and res['documents'][0]:
            for doc_text, meta in zip(res['documents'][0], res['metadatas'][0]):
                chunk_key = (doc_text, meta.get("source"), meta.get("page"))
                if chunk_key not in all_candidate_chunks:
                    all_candidate_chunks[chunk_key] = meta

    unique_chunks = list(all_candidate_chunks.keys())
    if not unique_chunks:
        raise HTTPException(status_code=404, detail="No matching chunks found.")

    # 3. Cross-Encoder Re-ranking against Original Query
    pair_inputs = [[payload.question, item[0]] for item in unique_chunks]
    scores = reranker.predict(pair_inputs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    selected_sources = []
    combined_context = []

    for rank, idx in enumerate(ranked_indices):
        chunk_text, doc_src, page_no = unique_chunks[idx]
        score = float(scores[idx])

        selected_sources.append(SourceDetail(
            doc=doc_src,
            page=page_no,
            score=round(score, 3),
            text=chunk_text
        ))
        combined_context.append(f"[Source {rank+1}: {doc_src} (Page {page_no})]\n{chunk_text}")

    context_str = "\n\n".join(combined_context)

    # 4. LLM Synthesis
    system_prompt = (
        "You are an enterprise research specialist. You have been supplied with verified context chunks "
        "retrieved using HyDE and Multi-Query fusion. Answer the question accurately."
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

    return HyDEResponse(
        session_id=session_id,
        original_question=payload.question,
        generated_hypothetical_doc=hypo_doc,
        expanded_queries=variations,
        sources=selected_sources,
        answer=answer
    )
