import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

app = FastAPI(title="Enterprise RAG Query Service", version="1.0.0")

# 1. Pipeline Initial Setup
pdf_path = r"C:\Users\BALU\OneDrive\Pictures\Screenshots\rag_project_prep\simple.pdf.pdf"
reader = PdfReader(pdf_path)

extracted_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        extracted_text += " " + text

words = extracted_text.split()
chunk_size = 50
chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day6_rag_collection")

for index, chunk in enumerate(chunks):
    vector = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[vector],
        ids=[f"chunk_{index}"]
    )

# 2. Gemini Client with Provided API Key
ai_client = genai.Client(api_key="AQ.Ab8RN6IQlaVp02xgDD4-3HlyM3HLqeJXCPLlpz9BnSNrTfOsTA")

# 3. Request & Response Data Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    retrieved_chunk: str
    answer: str

# 4. Endpoints
@app.get("/")
def home():
    return {"status": "Online", "service": "Enterprise RAG API", "total_indexed_chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vector = model.encode(payload.question).tolist()
    retrieval_results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    retrieved_chunk = retrieval_results['documents'][0][0]

    rag_prompt = f"""
    Answer the user's question using ONLY the retrieved context below.
    If the context does not contain the answer, reply that it is not available in the document.

    --- CONTEXT ---
    {retrieved_chunk}

    --- QUESTION ---
    {payload.question}
    """

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=rag_prompt,
        )
        answer_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=payload.question,
        retrieved_chunk=retrieved_chunk,
        answer=answer_text
    )