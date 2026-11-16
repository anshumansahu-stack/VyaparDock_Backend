from PDFExtractor_STEP_1 import extract_text_from_pdf
from TextCleaner_STEP_2 import clean_text


## Step 1: Extract the text from PDF
with open("pipeline/testcv/Resume Anshuman Updated.pdf", "rb") as f:
    file_bytes = f.read()

result = extract_text_from_pdf(file_bytes)
# print(result)

## Step 2: NLTK Text cleaning
result=clean_text(result)

print(result)
