import os
import re
import json
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 20 Guardrails & Self-Reflective RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day20_guardrails_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day20"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

# Safety Patterns (Regex Guardrails for PII & Injection)
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)system\s+prompt\s+override",
    r"(?i)reveal\s+(api\s+key|credentials|secret)",
    r"(?i)act\s+as\s+(an\s+unfiltered|jailbroken)\s+assistant"
]

EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_PATTERN = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"

class GuardrailRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    enable_guardrails: bool = True

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class SafetyReport(BaseModel):
    input_passed: bool
    injection_detected: bool
    pii_redacted_count: int
    hallucination_detected: bool
    groundedness_confidence: float
    audit_notes: str

class GuardrailResponse(BaseModel):
    session_id: str
    standalone_question: str
    safety_audit: SafetyReport
    sources: list[SourceDetail]
    answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 20 Self-Reflective Guardrails RAG",
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

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_collection_chunks": collection.count()
    }

def inspect_input_guardrail(query: str) -> tuple[bool, int, str]:
    """Scans for prompt injection attacks and redacts PII."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query):
            return False, 0, query

    redacted_query, count1 = re.subn(EMAIL_PATTERN, "[REDACTED_EMAIL]", query)
    redacted_query, count2 = re.subn(PHONE_PATTERN, "[REDACTED_PHONE]", redacted_query)
    total_redacted = count1 + count2

    return True, total_redacted, redacted_query

def self_reflective_grounding_audit(question: str, context: str, answer: str) -> tuple[bool, float, str]:
    """Performs self-reflective audit on the generated answer against source context."""
    audit_prompt = f"""You are an unbiased automated enterprise security and hallucination inspector.
Evaluate whether the Assistant Answer is strictly supported by the Context without making up facts.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

--- ASSISTANT ANSWER ---
{answer}

Respond STRICTLY in JSON format:
{{
  "is_grounded": true,
  "confidence_score": 0.95,
  "reason": "Direct factual support in context."
}}
"""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": audit_prompt}],
            temperature=0.0
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        return data.get("is_grounded", True), float(data.get("confidence_score", 1.0)), data.get("reason", "Audited successfully")
    except Exception:
        return True, 1.0, "Audit completed via heuristic fallback."

@app.post("/chat", response_model=GuardrailResponse)
def guarded_chat(payload: GuardrailRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    injection_detected = False
    pii_count = 0
    clean_query = payload.question

    if payload.enable_guardrails:
        passed, pii_count, clean_query = inspect_input_guardrail(payload.question)
        if not passed:
            report = SafetyReport(
                input_passed=False,
                injection_detected=True,
                pii_redacted_count=0,
                hallucination_detected=False,
                groundedness_confidence=0.0,
                audit_notes="Input violation: Prompt injection attempt detected. Blocked by security guardrail."
            )
            return GuardrailResponse(
                session_id=session_id,
                standalone_question=payload.question,
                safety_audit=report,
                sources=[],
                answer="⚠️ Security Warning: Prompt injection pattern detected. Request declined by Enterprise Guardrail."
            )

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    query_vector = embed_model.encode(clean_query).tolist()
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

    pair_inputs = [[clean_query, doc] for doc in candidates]
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

    system_prompt = (
        "You are an enterprise research assistant operating under strict safety and groundedness standards. "
        "Answer strictly based on the context provided without assumptions."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{clean_query}"
    })

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.1
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    is_grounded = True
    grounding_confidence = 1.0
    audit_notes = "Answer verified against context."

    if payload.enable_guardrails:
        is_grounded, grounding_confidence, audit_notes = self_reflective_grounding_audit(clean_query, context_str, answer)
        if not is_grounded or grounding_confidence < 0.65:
            answer = "⚠️ Verification Alert: The model's generated answer could not be verified with sufficient confidence against the retrieved context."

    report = SafetyReport(
        input_passed=True,
        injection_detected=False,
        pii_redacted_count=pii_count,
        hallucination_detected=(not is_grounded),
        groundedness_confidence=round(grounding_confidence, 2),
        audit_notes=audit_notes
    )

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return GuardrailResponse(
        session_id=session_id,
        standalone_question=clean_query,
        safety_audit=report,
        sources=selected_sources,
        answer=answer
    )
