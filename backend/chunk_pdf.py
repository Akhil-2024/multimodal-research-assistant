from pypdf import PdfReader
import json

pdf_path = "../data/pdfs/sample.pdf"
output_file = "chunks.json"

reader = PdfReader(pdf_path)

chunk_size = 800
overlap = 100

chunks = []

for page_num, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            "chunk_id": len(chunks),
            "page": page_num,
            "text": chunk
        })

        start = end - overlap

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)

print(f"Total chunks created: {len(chunks)}")
print(f"Saved to {output_file}")