import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 1. Ingestion: PDF nunchi text chunks prepare cheyadam
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
print(f"Total Chunks Prepared: {len(chunks)}")

# 2. Embedding Model Load Cheyadam
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Vector Database (ChromaDB) Setup
print("\nInitializing ChromaDB in-memory database...")
chroma_client = chromadb.Client()

# Kotha collection create chesthunnam (Tables laaga)
collection = chroma_client.get_or_create_collection(name="pdf_rag_knowledgebase")

# 4. Chunks ni Vector DB lo Store Cheyadam
print("Indexing Chunks into Vector Database...")
for index, chunk in enumerate(chunks):
    vector_embedding = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[vector_embedding],
        ids=[f"chunk_id_{index}"]
    )

print(f"Successfully Indexed {collection.count()} chunks into ChromaDB.")

# 5. Semantic Search Testing (User Question Search)
user_query = "What is this document about?"
print(f"\nSearching for Query: '{user_query}'")

query_vector = model.encode(user_query).tolist()
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1  # Top 1 most relevant chunk
)

print("\n--- Semantic Search Result ---")
print(f"Best Matching Chunk: {results['documents'][0][0]}")
print(f"Distance/Similarity Score: {results['distances'][0][0]}")