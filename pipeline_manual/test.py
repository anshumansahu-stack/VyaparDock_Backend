from step1_pdfExtractor import extract_text_from_pdf


## Step 1: Extract the text from PDF
with open("pipeline/testcv/23CHB0A03-CV.pdf", "rb") as f:
    file_bytes = f.read()

result = extract_text_from_pdf(file_bytes)
print(result)
