# ==========================================
# 1. Variables & Types (Data Types)
# ==========================================
project_name = "Enterprise RAG Platform"  # String (Text)
target_accuracy = 95.5                   # Float (Decimals)
is_production_ready = False              # Boolean (True/False)
total_pages = 12                         # Integer (Number)

print(f"Project: {project_name}")
print(f"Target Accuracy: {target_accuracy}% | Pages to process: {total_pages}\n")

# ==========================================
# 2. List (Document Chunks list la store avthayi)
# ==========================================
# RAG lo PDF ni chinna text blocks ga cut chestham, vatine 'chunks' antam
chunks = [
    "Machine learning enables systems to learn from data.",
    "FastAPI is a modern web framework for Python.",
    "ChromaDB is a vector database used for AI search."
]

print("--- Extracted Chunks ---")
for index, chunk in enumerate(chunks, start=1):
    print(f"Chunk {index}: {chunk}")

# ==========================================
# 3. Dictionary (Key-Value: Metadata store cheyadaniki)
# ==========================================
# Prathi chunk ki page number and filename attach cheyali
chunk_metadata = {
    "file_name": "ai_research_paper.pdf",
    "page_number": 3,
    "chunk_id": "chunk_003",
    "word_count": 85
}

print("\n--- Metadata Preview ---")
print(f"Source File: {chunk_metadata['file_name']}")
print(f"Located on Page: {chunk_metadata['page_number']}")

# ==========================================
# 4. Functions & Conditions (Simple Logic)
# ==========================================
def check_chunk_validity(word_count):
    # Minimum 20 words unte ne adi valid chunk
    if word_count >= 20:
        return "Valid chunk for vector embedding"
    else:
        return "Too short, discard this chunk"

result = check_chunk_validity(chunk_metadata["word_count"])
print(f"Status: {result}")