from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 1. Day 2 nunchi Text Chunks theeskovadam
pdf_path = r"C:\Users\BALU\OneDrive\Pictures\Screenshots\rag_project_prep\simple.pdf.pdf"
reader = PdfReader(pdf_path)

extracted_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        extracted_text += " " + text

# Word chunking (50 words per chunk)
words = extracted_text.split()
chunk_size = 50
chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

print(f"Total Text Chunks Prepared: {len(chunks)}")

# 2. Embedding Model Load Cheyadam
print("\nLoading Embedding Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Chunks ni Vector Embeddings ga convert cheyadam
print("Generating Vector Embeddings for Chunks...")
embeddings = model.encode(chunks)

# 4. Results & Vector Shapes Preview
print(f"\n--- Embedding Pipeline Summary ---")
print(f"Total Embeddings Generated: {len(embeddings)}")
print(f"Embedding Vector Dimensions: {len(embeddings[0])} (Numbers per chunk)")
print(f"\nPreview of Chunk 1 Vector (First 5 numbers):")
print(embeddings[0][:5])