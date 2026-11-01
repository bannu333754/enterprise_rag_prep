import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

# 1. Ingestion & Chunking
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

# 2. Embedding Model & Vector DB
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day5_rag_collection")

for index, chunk in enumerate(chunks):
    vector = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[vector],
        ids=[f"chunk_{index}"]
    )

# 3. User Query & Semantic Retrieval
user_query = "What are the main topics discussed in this document?"
print(f"User Query: {user_query}")

query_vector = model.encode(user_query).tolist()
retrieval_results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
)

retrieved_context = retrieval_results['documents'][0][0]
print(f"\nRetrieved Context from PDF:\n{retrieved_context}\n")

# 4. Augmented Prompt Assembly
rag_prompt = f"""
You are a factual assistant. Answer the user's question using ONLY the retrieved context below.
If the context does not contain the answer, state that the information is not in the document.

--- RETRIEVED CONTEXT ---
{retrieved_context}

--- USER QUESTION ---
{user_query}
"""

# 5. LLM Generation
print("Generating final grounded answer via LLM...")
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=rag_prompt,
)

print("\n--- FINAL RAG GENERATED RESPONSE ---")
print(response.text)