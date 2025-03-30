import requests
import fitz

def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_path = "downloaded.pdf"

    # Save PDF locally
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    # Extract text
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])

    return text
