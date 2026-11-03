import os
import json
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 22 Iterative Self-Corrective RAG API",
    version="1.0.0"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day22_iterative_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day22"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class IterativeRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class IterationStep(BaseModel):
    iteration_number: int
    query_used: str
    retrieved_chunks_count: int
    evaluation_confidence: float
    needs_retrieval: bool
    missing_info_gaps: str

class IterativeResponse(BaseModel):
    session_id: str
    standalone_question: str
    total_iterations: int
    iteration_history: List[IterationStep]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 22 Iterative Self-Corrective RAG",
        "chunks": collection.count(),
        "sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")

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
        chunk_size = 75
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

    return {"status": "Success", "filename": file.filename, "chunks_indexed": total_added}

def fetch_chunks_for_query(query: str, target_doc: str = "ALL") -> List[Dict]:
    query_vector = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        return []

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    returned_docs = []
    for i in ranked_idx:
        returned_docs.append({
            "text": candidates[i],
            "source": metas[i].get("source", "Unknown"),
            "page": metas[i].get("page", 0)
        })
    return returned_docs

def evaluate_answer_and_detect_gaps(question: str, context: str, draft: str) -> Dict:
    eval_prompt = f"""You are an expert RAG quality inspector.
Question: {question}
Context: {context}
Draft Answer: {draft}

Assess if the draft fully answers the question based on the context.
Return STRICTLY JSON:
{{
  "confidence": 0.5,
  "needs_retrieval": true,
  "missing_gaps": "Explain what key detail is missing",
  "supplementary_query": "New specific query to find missing data"
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception:
        return {
            "confidence": 0.85,
            "needs_retrieval": False,
            "missing_gaps": "None",
            "supplementary_query": ""
        }

def generate_answer(question: str, context_str: str, history: List) -> str:
    msgs = [{"role": "system", "content": "You are an advanced enterprise analyst. Answer accurately using the grounded context."}]
    for m in history[-4:]:
        msgs.append(m)
    msgs.append({"role": "user", "content": f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{question}"})

    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=msgs,
        temperature=0.2
    )
    return res.choices[0].message.content

@app.post("/chat", response_model=IterativeResponse)
def iterative_chat(payload: IterativeRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    iteration_traces: List[IterationStep] = []
    all_chunks = []
    current_query = payload.question
    max_iterations = 2
    draft = ""

    for iter_num in range(1, max_iterations + 1):
        new_docs = fetch_chunks_for_query(current_query, payload.target_doc)
        all_chunks.extend(new_docs)

        unique_chunks = {c.get('text'): c for c in all_chunks}.values()
        context_blocks = [f"[{c.get('source')} P. {c.get('page')}]: {c.get('text')}" for c in unique_chunks]
        context_str = "\n\n".join(context_blocks)

        draft = generate_answer(payload.question, context_str, history)
        eval_res = evaluate_answer_and_detect_gaps(payload.question, context_str, draft)

        iteration_traces.append(IterationStep(
            iteration_number=iter_num,
            query_used=current_query,
            retrieved_chunks_count=len(new_docs),
            evaluation_confidence=round(eval_res.get("confidence", 1.0), 2),
            needs_retrieval=eval_res.get("needs_retrieval", False),
            missing_info_gaps=eval_res.get("missing_gaps", "None")
        ))

        if not eval_res.get("needs_retrieval", False) or iter_num == max_iterations:
            break

        current_query = eval_res.get("supplementary_query") or payload.question

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": draft})

    return IterativeResponse(
        session_id=session_id,
        standalone_question=payload.question,
        total_iterations=len(iteration_traces),
        iteration_history=iteration_traces,
        final_answer=draft
    )
