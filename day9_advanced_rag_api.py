import os
import shutil
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 9 Enterprise RAG with Re-ranking", version="1.0.0")

# Embedding model, Cross-Encoder Re-ranker, and ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day9_rerank_rag")

# Day 8 Groq working key
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day9"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    target_doc: str = "ALL"

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class QueryResponse(BaseModel):
    question: str
    target_doc: str
    sources: list[SourceDetail]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 9 Re-ranking RAG API",
        "total_chunks": collection.count()
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
                metadatas=[{
                    "source": file.filename,
                    "page": page_idx + 1
                }],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_collection_chunks": collection.count()
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    query_vector = embed_model.encode(payload.question).tolist()

    where_filter = None
    if payload.target_doc and payload.target_doc != "ALL":
        where_filter = {"source": payload.target_doc}

    # Retrieve candidate chunks (Top-4)
    n_candidates = min(4, collection.count())
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_candidates,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Re-ranking using Cross-Encoder
    pair_inputs = [[payload.question, doc_text] for doc_text in candidates]
    scores = reranker.predict(pair_inputs)

    # Rank and select Top-2 chunks based on highest Cross-Encoder relevance
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

        combined_context.append(
            f"[Source {rank+1}: {meta.get('source')} (Page {meta.get('page')})]\n{chunk_text}"
        )

    context_str = "\n\n".join(combined_context)

    rag_prompt = f"""
Answer the user's question precisely using the verified context blocks below.
If the context does not contain the answer, say that it is not available in the document.

--- CONTEXT ---
{context_str}

--- QUESTION ---
{payload.question}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a precise enterprise RAG assistant with re-ranked grounding."},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.2,
        )
        answer_text = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=payload.question,
        target_doc=payload.target_doc,
        sources=selected_sources,
        answer=answer_text
    )