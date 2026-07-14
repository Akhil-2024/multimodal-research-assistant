from pypdf import PdfReader

pdf_path = "../data/pdfs/sample.pdf"

reader = PdfReader(pdf_path)

print(f"Total Pages: {len(reader.pages)}\n")

for i, page in enumerate(reader.pages):

    text = page.extract_text()

    print(f"===== PAGE {i+1} =====")
    print(text[:1000])
    print("\n")