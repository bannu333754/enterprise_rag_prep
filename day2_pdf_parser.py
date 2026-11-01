from pypdf import PdfReader

# Exact absolute path tho load chesthunnam (no path error)
pdf_path = r"C:\Users\BALU\OneDrive\Pictures\Screenshots\rag_project_prep\simple.pdf.pdf"
reader = PdfReader(pdf_path)

print(f"Total Pages Found: {len(reader.pages)}")

# Extract Text Page by Page
extracted_text = ""
for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    if page_text:
        extracted_text += f"\n--- Page {page_number} ---\n" + page_text

print("\n--- Extracted Text Sample (First 300 chars) ---")
print(extracted_text[:300])

# Simple Word-based Chunker (RAG Core Logic)
words = extracted_text.split()
chunk_size = 50  # Prathi chunk lo 50 words
chunks = []

for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size])
    chunks.append(chunk)

print(f"\nTotal Chunks Created: {len(chunks)}")
if chunks:
    print(f"Preview Chunk 1:\n{chunks[0]}")