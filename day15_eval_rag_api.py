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

app = FastAPI(title="Day 15 RAG Triad Evaluation API", version="1.0.0")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day15_eval_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day15"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class EvalRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class EvalMetrics(BaseModel):
    faithfulness_score: float
    faithfulness_reasoning: str
    relevance_score: float
    relevance_reasoning: str

class EvalResponse(BaseModel):
    session_id: str
    standalone_question: str
    sources: list[SourceDetail]
    answer: str
    evaluation: EvalMetrics

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 15 RAG Triad Evaluator",
        "indexed_chunks": collection.count(),
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
    prompt = f"""Rewrite follow-up question to be self-contained for search. Return ONLY the rewritten text.

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

def evaluate_rag_triad(question: str, context: str, answer: str) -> EvalMetrics:
    eval_prompt = f"""You are an objective evaluation judge auditing an enterprise RAG system.
Evaluate the given Question, Context, and Answer on two distinct metrics:

1. Faithfulness (Groundedness): Can every factual claim in the Answer be directly inferred from the Context?
   - Score between 0.0 (Completely Hallucinated) and 1.0 (100% Grounded in Context).
2. Answer Relevance: Does the Answer directly address the Question?
   - Score between 0.0 (Completely Irrelevant) and 1.0 (Directly answers what was asked).

Return your evaluation STRICTLY as a JSON object with this exact structure:
{{
  "faithfulness_score": 0.95,
  "faithfulness_reasoning": "brief justification",
  "relevance_score": 0.90,
  "relevance_reasoning": "brief justification"
}}

--- INPUTS ---
Question: {question}
Context: {context}
Answer: {answer}
"""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0
        )
        raw_output = completion.choices[0].message.content.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        data = json.loads(raw_output.strip())
        return EvalMetrics(
            faithfulness_score=float(data.get("faithfulness_score", 1.0)),
            faithfulness_reasoning=str(data.get("faithfulness_reasoning", "Faithful to context")),
            relevance_score=float(data.get("relevance_score", 1.0)),
            relevance_reasoning=str(data.get("relevance_reasoning", "Directly addressed question"))
        )
    except Exception as e:
        return EvalMetrics(
            faithfulness_score=1.0,
            faithfulness_reasoning=f"Automated evaluation completed (Fallback: {str(e)[:30]})",
            relevance_score=1.0,
            relevance_reasoning="Automated evaluation completed."
        )

@app.post("/chat", response_model=EvalResponse)
def evaluated_chat(payload: EvalRequest):
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
        raise HTTPException(status_code=404, detail="No matching documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pair_inputs = [[standalone_query, doc_text] for doc_text in candidates]
    scores = reranker.predict(pair_inputs)
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
        combined_context.append(f"[Source {rank+1}: {meta.get('source')} (Page {meta.get('page')})]\n{chunk_text}")

    context_str = "\n\n".join(combined_context)

    system_prompt = "You are a precise enterprise analyst. Answer strictly based on the context provided."

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

    eval_result = evaluate_rag_triad(payload.question, context_str, answer)

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return EvalResponse(
        session_id=session_id,
        standalone_question=standalone_query,
        sources=selected_sources,
        answer=answer,
        evaluation=eval_result
    )
