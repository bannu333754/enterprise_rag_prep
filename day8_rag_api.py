import os
import shutil
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Enterprise Multi-Doc Dynamic RAG", version="2.0.0")

# Embedding model and ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day8_multidoc_rag")

# Groq client initialization
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    target_doc: str = "ALL"

class QueryResponse(BaseModel):
    question: str
    target_doc: str
    retrieved_chunk: str
    source_doc: str
    page_num: int
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Multi-Document Dynamic RAG API (Groq)",
        "total_chunks_indexed": collection.count()
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
        chunk_size = 50
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
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    query_vector = embed_model.encode(payload.question).tolist()

    where_filter = None
    if payload.target_doc and payload.target_doc != "ALL":
        where_filter = {"source": payload.target_doc}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No relevant context found matching the criteria.")

    retrieved_chunk = results['documents'][0][0]
    meta = results['metadatas'][0][0]
    source_doc = meta.get("source", "Unknown")
    page_num = meta.get("page", 0)

    rag_prompt = f"""
Answer the user's question using ONLY the retrieved context below.
If the context does not contain the answer, reply that it is not available in the document.

--- CONTEXT ---
Source: {source_doc} (Page {page_num})
{retrieved_chunk}

--- QUESTION ---
{payload.question}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a precise enterprise RAG assistant. Answer strictly based on the provided context."},
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
        retrieved_chunk=retrieved_chunk,
        source_doc=source_doc,
        page_num=page_num,
        answer=answer_text
    )