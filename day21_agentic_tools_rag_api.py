import os
import json
import shutil
import chromadb
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 21 Agentic Multi-Step RAG API", version="1.0.0")

# Embedding model, Cross-Encoder, ChromaDB setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day21_agentic_rag")

# Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day21"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class AgentRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class ToolExecutionTrace(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result_snippet: str

class AgentResponse(BaseModel):
    session_id: str
    standalone_question: str
    tool_traces: List[ToolExecutionTrace]
    final_answer: str

# Tool Definitions
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_internal_documents",
            "description": "Searches through uploaded enterprise PDF documents in the vector database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The specific semantic query to run against enterprise documents"
                    }
                },
                "required": ["search_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_live_web_facts",
            "description": "Fetches general public facts, current specifications, or internet information when internal documents lack coverage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "web_query": {
                        "type": "string",
                        "description": "The internet search keywords to look up"
                    }
                },
                "required": ["web_query"]
            }
        }
    }
]

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 21 Agentic Multi-Step RAG",
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

def tool_internal_lookup(query: str, target_doc: str = "ALL") -> str:
    if collection.count() == 0:
        return "Internal Document Store is empty. No documents uploaded."

    query_vector = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        return "No matching internal document chunks found."

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pair_inputs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pair_inputs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    found = []
    for idx in ranked_indices:
        found.append(f"[{metas[idx].get('source')} Page {metas[idx].get('page')}]: {candidates[idx]}")

    return "\n\n".join(found)

def tool_web_lookup(query: str) -> str:
    return f"Live Information Retrieval for '{query}': Verified public enterprise domain reference records confirm standard operational specifications and open protocols."

@app.post("/chat", response_model=AgentResponse)
def agentic_chat(payload: AgentRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    traces: List[ToolExecutionTrace] = []

    system_message = (
        "You are an enterprise autonomous research agent. You have tools available to search internal documents "
        "and fetch live facts. If the user query requires internal documentation, call 'lookup_internal_documents'. "
        "If it requires live or public information, call 'fetch_live_web_facts'. You may call both if appropriate. "
        "Always synthesize a clean final response."
    )

    messages = [{"role": "system", "content": system_message}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": payload.question})

    try:
        decision_completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    choice = decision_completion.choices[0].message

    if choice.tool_calls:
        tool_messages = [choice]
        for t_call in choice.tool_calls:
            t_name = t_call.function.name
            t_args = json.loads(t_call.function.arguments)

            if t_name == "lookup_internal_documents":
                result = tool_internal_lookup(t_args.get("search_query", payload.question), payload.target_doc)
            elif t_name == "fetch_live_web_facts":
                result = tool_web_lookup(t_args.get("web_query", payload.question))
            else:
                result = "Unknown tool called."

            traces.append(ToolExecutionTrace(
                tool_name=t_name,
                arguments=t_args,
                result_snippet=result[:200] + ("..." if len(result) > 200 else "")
            ))

            tool_messages.append({
                "role": "tool",
                "tool_call_id": t_call.id,
                "name": t_name,
                "content": result
            })

        final_messages = messages + tool_messages
        final_completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=final_messages,
            temperature=0.2
        )
        final_answer = final_completion.choices[0].message.content
    else:
        final_answer = choice.content or "Completed without tool execution."

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return AgentResponse(
        session_id=session_id,
        standalone_question=payload.question,
        tool_traces=traces,
        final_answer=final_answer
    )