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

app = FastAPI(title="Day 24 Adaptive Complexity RAG API", version="1.0.0")

# Embedding & CrossEncoder Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day24_adaptive_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day24"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class AdaptiveRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class ComplexityDecision(BaseModel):
    tier: str  # "NO_RETRIEVAL", "SINGLE_PASS", "MULTI_HOP"
    reasoning: str
    sub_queries: List[str]

class RetrievedItem(BaseModel):
    source: str
    page: int
    score: float
    text: str

class AdaptiveResponse(BaseModel):
    session_id: str
    standalone_question: str
    decision: ComplexityDecision
    retrieved_sources: List[RetrievedItem]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 24 Adaptive Complexity RAG",
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

def analyze_complexity(query: str) -> ComplexityDecision:
    """Classifies user query into NO_RETRIEVAL, SINGLE_PASS, or MULTI_HOP strategies."""
    classifier_prompt = f"""You are an enterprise query routing classifier.
Analyze the user question and classify its retrieval requirements into one of three tiers:
- "NO_RETRIEVAL": Greetings, casual remarks, general logic/programming, or conversational pleasantries that do not need company document retrieval.
- "SINGLE_PASS": Simple, direct factual questions requiring a single document lookup.
- "MULTI_HOP": Comparative, multi-step, or analytical questions requiring multi-query breakdown or cross-referencing.

User Question: {query}

Respond strictly in JSON:
{{
  "tier": "NO_RETRIEVAL | SINGLE_PASS | MULTI_HOP",
  "reasoning": "Brief justification",
  "sub_queries": ["sub query 1", "sub query 2"]
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        parsed = json.loads(raw.strip())
        return ComplexityDecision(
            tier=parsed.get("tier", "SINGLE_PASS"),
            reasoning=parsed.get("reasoning", "Default classified"),
            sub_queries=parsed.get("sub_queries", [query])
        )
    except Exception:
        return ComplexityDecision(
            tier="SINGLE_PASS",
            reasoning="Fallback heuristic",
            sub_queries=[query]
        )

def execute_retrieval(queries: List[str], target_doc: str = "ALL", limit_per_query: int = 3) -> List[RetrievedItem]:
    """Retrieves and re-ranks passages across one or more query variations."""
    if collection.count() == 0:
        return []

    collected_chunks = {}
    where_filter = {"source": target_doc} if target_doc != "ALL" else None

    for q in queries:
        q_vec = embed_model.encode(q).tolist()
        n_res = min(limit_per_query, collection.count())
        res = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)

        if res['documents'] and res['documents'][0]:
            docs = res['documents'][0]
            metas = res['metadatas'][0]
            for d, m in zip(docs, metas):
                collected_chunks[d] = m

    if not collected_chunks:
        return []

    unique_docs = list(collected_chunks.keys())
    primary_query = queries[0]
    pairs = [[primary_query, doc] for doc in unique_docs]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    retrieved_items = []
    for idx in ranked_indices:
        doc_text = unique_docs[idx]
        meta = collected_chunks[doc_text]
        retrieved_items.append(RetrievedItem(
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            score=round(float(scores[idx]), 3),
            text=doc_text
        ))

    return retrieved_items

@app.post("/chat", response_model=AdaptiveResponse)
def adaptive_chat(payload: AdaptiveRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Query Complexity Routing
    decision = analyze_complexity(payload.question)
    retrieved_sources: List[RetrievedItem] = []
    context_str = ""

    # Step 2: Adaptive Strategy Execution
    if decision.tier == "NO_RETRIEVAL":
        system_prompt = "You are an intelligent enterprise AI assistant. Answer the user conversational prompt directly and naturally."
    elif decision.tier == "SINGLE_PASS":
        retrieved_sources = execute_retrieval([payload.question], payload.target_doc, limit_per_query=4)
        context_blocks = [f"[{s.source} Page {s.page}]: {s.text}" for s in retrieved_sources]
        context_str = "\n\n".join(context_blocks)
        system_prompt = "You are an enterprise research assistant. Answer the question directly using the provided context."
    else:  # MULTI_HOP
        search_queries = decision.sub_queries if decision.sub_queries else [payload.question]
        retrieved_sources = execute_retrieval(search_queries, payload.target_doc, limit_per_query=3)
        context_blocks = [f"[{s.source} Page {s.page}]: {s.text}" for s in retrieved_sources]
        context_str = "\n\n".join(context_blocks)
        system_prompt = "You are an advanced enterprise analytical assistant. Synthesize a comprehensive comparative answer across multiple contextual points."

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if context_str:
        user_content = f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    else:
        user_content = payload.question

    messages.append({"role": "user", "content": user_content})

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        final_answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return AdaptiveResponse(
        session_id=session_id,
        standalone_question=payload.question,
        decision=decision,
        retrieved_sources=retrieved_sources,
        final_answer=final_answer
    )
