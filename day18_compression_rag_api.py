import os
import re
import shutil
import chromadb
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 18 Contextual Compression RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day18_compression_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day18"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}
SENTENCE_SCORE_THRESHOLD = 0.10

class CompressionRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    enable_compression: bool = True

class CompressedSource(BaseModel):
    doc: str
    page: int
    original_length_words: int
    compressed_length_words: int
    compression_ratio_pct: float
    extracted_sentences: list[str]

class CompressionResponse(BaseModel):
    session_id: str
    standalone_question: str
    total_raw_words: int
    total_compressed_words: int
    net_savings_pct: float
    sources: list[CompressedSource]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 18 Contextual Compression RAG",
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
        chunk_size = 80
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
    prompt = f"""Rewrite the follow-up question to be self-contained for search. Return ONLY the query text.
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

def compress_chunk_by_sentences(query: str, raw_text: str) -> List[str]:
    """Splits chunk into sentences and keeps only those passing semantic cross-encoder threshold."""
    # Split by periods, exclamation, or question marks
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if len(s.strip().split()) > 3]
    if not sentences:
        return [raw_text]

    pairs = [[query, s] for s in sentences]
    scores = reranker.predict(pairs)

    retained = [sentences[i] for i, score in enumerate(scores) if score >= SENTENCE_SCORE_THRESHOLD]
    
    # Fallback to the single highest-scoring sentence if none met the threshold
    if not retained:
        best_idx = int(scores.argmax())
        retained = [sentences[best_idx]]

    return retained

@app.post("/chat", response_model=CompressionResponse)
def compressed_chat(payload: CompressionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

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
        raise HTTPException(status_code=404, detail="No documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Re-rank candidates to pick top 2 chunks
    pair_inputs = [[standalone_query, doc] for doc in candidates]
    scores = reranker.predict(pair_inputs)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    compressed_sources = []
    final_context_blocks = []
    total_raw_words = 0
    total_compressed_words = 0

    for rank, idx in enumerate(top_indices):
        raw_chunk = candidates[idx]
        meta = metas[idx]
        raw_word_count = len(raw_chunk.split())
        total_raw_words += raw_word_count

        if payload.enable_compression:
            extracted_sentences = compress_chunk_by_sentences(standalone_query, raw_chunk)
            compressed_chunk = " ".join(extracted_sentences)
        else:
            extracted_sentences = [raw_chunk]
            compressed_chunk = raw_chunk

        comp_word_count = len(compressed_chunk.split())
        total_compressed_words += comp_word_count

        savings = round(((raw_word_count - comp_word_count) / raw_word_count) * 100, 1) if raw_word_count > 0 else 0.0

        compressed_sources.append(CompressedSource(
            doc=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            original_length_words=raw_word_count,
            compressed_length_words=comp_word_count,
            compression_ratio_pct=savings,
            extracted_sentences=extracted_sentences
        ))

        final_context_blocks.append(f"[Source {rank+1}: {meta.get('source')} (Page {meta.get('page')})]\n{compressed_chunk}")

    net_savings = round(((total_raw_words - total_compressed_words) / total_raw_words) * 100, 1) if total_raw_words > 0 else 0.0
    context_str = "\n\n".join(final_context_blocks)

    system_prompt = (
        "You are an enterprise research assistant provided with surgically compressed, high-density context blocks. "
        "Answer the user question directly and concisely using the compressed facts."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- COMPRESSED CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
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

    return CompressionResponse(
        session_id=session_id,
        standalone_question=standalone_query,
        total_raw_words=total_raw_words,
        total_compressed_words=total_compressed_words,
        net_savings_pct=net_savings,
        sources=compressed_sources,
        answer=answer
    )
