import os
import json
import shutil
import chromadb
import networkx as nx
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 16 GraphRAG API", version="1.0.0")

# Vector Database & Cross-Encoder setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day16_graph_rag")

# Working Groq Client
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day16"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-Memory Knowledge Graph via NetworkX
KG = nx.DiGraph()
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class GraphRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class SourceDetail(BaseModel):
    doc: str
    page: int
    score: float
    text: str

class GraphResponse(BaseModel):
    session_id: str
    standalone_question: str
    extracted_entities: list[str]
    graph_triplets: list[str]
    sources: list[SourceDetail]
    answer: str

def extract_triplets_from_text(text_chunk: str) -> List[Dict[str, str]]:
    """Uses LLM to extract entity-relation triples (Head, Relation, Tail)."""
    prompt = f"""Extract key knowledge graph relationships from the text below.
Identify relationships in the format: Subject -> Relation -> Object.

Respond STRICTLY in a JSON array format like:
[
  {{"head": "Alice", "relation": "founded", "tail": "TechCorp"}},
  {{"head": "TechCorp", "relation": "located_in", "tail": "Hyderabad"}}
]
If no clear relationships exist, respond with [].

Text:
{text_chunk[:1500]}
"""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = completion.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception:
        return []

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 16 GraphRAG (Knowledge Graph + Vector RAG)",
        "graph_nodes": KG.number_of_nodes(),
        "graph_edges": KG.number_of_edges(),
        "total_vector_chunks": collection.count()
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
    triplets_count = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 70
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

            # Extract Triplets for Knowledge Graph
            triplets = extract_triplets_from_text(chunk)
            for t in triplets:
                h = str(t.get("head", "")).strip().lower()
                r = str(t.get("relation", "")).strip().lower()
                tail = str(t.get("tail", "")).strip().lower()
                if h and tail and r:
                    KG.add_node(h, label="Entity")
                    KG.add_node(tail, label="Entity")
                    KG.add_edge(h, tail, relation=r, source=file.filename)
                    triplets_count += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "graph_triplets_created": triplets_count,
        "total_nodes": KG.number_of_nodes(),
        "total_edges": KG.number_of_edges()
    }

def rewrite_query(history: List[Dict[str, str]], question: str) -> str:
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"""Rewrite follow-up question to be self-contained. Return ONLY the rewritten text.
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

def query_knowledge_graph(query_text: str) -> tuple[List[str], List[str]]:
    """Identifies nodes matching tokens in query and discovers 1-hop & 2-hop graph paths."""
    matched_nodes = []
    triplet_strings = []
    q_tokens = query_text.lower().split()

    for node in KG.nodes():
        for token in q_tokens:
            if len(token) > 3 and token in node:
                matched_nodes.append(node)
                break

    matched_nodes = list(set(matched_nodes))[:5]

    for node in matched_nodes:
        # Outgoing edges
        for _, neighbor, data in KG.out_edges(node, data=True):
            rel = data.get("relation", "connected_to")
            triplet_strings.append(f"({node.title()}) -[{rel}]-> ({neighbor.title()})")
        # Incoming edges
        for predecessor, _, data in KG.in_edges(node, data=True):
            rel = data.get("relation", "connected_to")
            triplet_strings.append(f"({predecessor.title()}) -[{rel}]-> ({node.title()})")

    return matched_nodes, list(set(triplet_strings))[:8]

@app.post("/chat", response_model=GraphResponse)
def graph_rag_chat(payload: GraphRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0 and KG.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    standalone_query = rewrite_query(history, payload.question)

    # 1. Knowledge Graph Traversal
    entities, graph_paths = query_knowledge_graph(standalone_query)

    # 2. Vector Retrieval & Re-ranking
    query_vector = embed_model.encode(standalone_query).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None

    n_candidates = min(4, collection.count()) if collection.count() > 0 else 0
    selected_sources = []
    context_chunks = []

    if n_candidates > 0:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_candidates,
            where=where_filter
        )
        if results['documents'] and results['documents'][0]:
            candidates = results['documents'][0]
            metas = results['metadatas'][0]

            pair_inputs = [[standalone_query, doc] for doc in candidates]
            scores = reranker.predict(pair_inputs)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

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
                context_chunks.append(f"[Source {rank+1}: {meta.get('source')} (Page {meta.get('page')})]\n{chunk_text}")

    # 3. Combine Vector Context + Knowledge Graph Triples
    graph_context_str = "\n".join(graph_paths) if graph_paths else "No direct entity relations traversed."
    vector_context_str = "\n\n".join(context_chunks) if context_chunks else "No vector chunks found."

    combined_prompt = f"""--- KNOWLEDGE GRAPH RELATIONS (Structured Facts) ---
{graph_context_str}

--- RETRIEVED DOCUMENT TEXT (Unstructured Chunks) ---
{vector_context_str}

--- QUESTION ---
{payload.question}"""

    system_prompt = (
        "You are an enterprise GraphRAG reasoning specialist. "
        "Synthesize facts by combining structured Knowledge Graph relationships with document context. "
        "Explain multi-entity connections clearly when relevant."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": combined_prompt})

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

    return GraphResponse(
        session_id=session_id,
        standalone_question=standalone_query,
        extracted_entities=entities,
        graph_triplets=graph_paths,
        sources=selected_sources,
        answer=answer
    )
