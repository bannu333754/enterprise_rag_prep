import os
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 25 Speculative Dual-Stage RAG API", version="1.0.0")

# Embedding & CrossEncoder Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day25_speculative_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day25"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class SpeculativeRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class DraftCandidate(BaseModel):
    subset_id: int
    draft_text: str
    evidence_source: str

class VerificationResult(BaseModel):
    verified_answer: str
    confidence_score: float
    verification_rationale: str

class SpeculativeResponse(BaseModel):
    session_id: str
    standalone_question: str
    drafts: List[DraftCandidate]
    verification: VerificationResult

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 25 Speculative RAG",
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

def retrieve_and_split_subsets(query: str, target_doc: str = "ALL"):
    """Retrieves top chunks and splits them into distinct evidence subsets for speculative drafting."""
    if collection.count() == 0:
        return []

    q_vec = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        return []

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    items = []
    for idx in ranked_indices:
        items.append({
            "text": candidates[idx],
            "source": f"{metas[idx].get('source')} Page {metas[idx].get('page')}"
        })
    return items

def generate_speculative_draft(subset_id: int, evidence: Dict, question: str) -> str:
    """Generates an initial fast draft candidate conditioned strictly on an evidence subset."""
    prompt = f"""You are a fast speculative drafter. Draft a direct, factual answer based ONLY on the evidence below.

Evidence ({evidence['source']}):
{evidence['text']}

Question: {question}

Draft:"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Draft generation error: {str(e)}"

def verify_and_synthesize(question: str, drafts: List[DraftCandidate], full_context: str) -> VerificationResult:
    """High-precision verification stage: audits drafts and produces a grounded synthesis."""
    drafts_text = "\n\n".join([f"Candidate Draft {d.subset_id} (from {d.evidence_source}):\n{d.draft_text}" for d in drafts])
    
    verifier_prompt = f"""You are an enterprise verification specialist. 
Evaluate the candidate drafts against the grounded context. 
Verify facts, eliminate hallucinations or contradictory claims, and synthesize the final accurate answer.

--- GROUNDED CONTEXT ---
{full_context}

--- CANDIDATE DRAFTS ---
{drafts_text}

--- QUESTION ---
{question}

Provide your response in the following format:
FINAL_ANSWER: <Your consolidated, factual answer>
RATIONALE: <Brief explanation of what was verified or corrected>
CONFIDENCE: <Confidence score between 0.0 and 1.0>"""

    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": verifier_prompt}],
            temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        
        final_ans = ""
        rationale = "Verified against primary context"
        conf = 0.95

        lines = content.splitlines()
        capturing_answer = False
        ans_lines = []

        for line in lines:
            if line.startswith("FINAL_ANSWER:"):
                capturing_answer = True
                ans_lines.append(line.replace("FINAL_ANSWER:", "").strip())
            elif line.startswith("RATIONALE:"):
                capturing_answer = False
                rationale = line.replace("RATIONALE:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                capturing_answer = False
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                except Exception:
                    conf = 0.95
            elif capturing_answer:
                ans_lines.append(line)

        final_ans = "\n".join(ans_lines).strip() if ans_lines else content

        return VerificationResult(
            verified_answer=final_ans,
            confidence_score=conf,
            verification_rationale=rationale
        )
    except Exception as e:
        return VerificationResult(
            verified_answer=drafts[0].draft_text if drafts else "Verification failed.",
            confidence_score=0.5,
            verification_rationale=f"Verifier error fallback: {str(e)}"
        )

@app.post("/chat", response_model=SpeculativeResponse)
def speculative_chat(payload: SpeculativeRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Retrieval & Subsets
    evidence_items = retrieve_and_split_subsets(payload.question, payload.target_doc)
    if not evidence_items:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    # Step 2: Speculative Drafting
    drafts: List[DraftCandidate] = []
    for idx, item in enumerate(evidence_items[:2]):
        draft_text = generate_speculative_draft(idx + 1, item, payload.question)
        drafts.append(DraftCandidate(
            subset_id=idx + 1,
            draft_text=draft_text,
            evidence_source=item['source']
        ))

    # Step 3: Verification & Synthesis
    full_context = "\n\n".join([f"[{it['source']}]: {it['text']}" for it in evidence_items])
    verification = verify_and_synthesize(payload.question, drafts, full_context)

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": verification.verified_answer})

    return SpeculativeResponse(
        session_id=session_id,
        standalone_question=payload.question,
        drafts=drafts,
        verification=verification
    )
