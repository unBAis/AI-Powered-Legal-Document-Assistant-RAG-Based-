
import requests
from bs4 import BeautifulSoup
import fitz
import json
import os
import pandas as pd
import sys
# Get the absolute path of the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.web_scrape import extract_text_from_pdf


url = "https://www.sci.gov.in/judgements-case-no/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract judgment links
for link in soup.find_all("a", href=True):
    if "pdf" in link["href"]:
        print("Found judgment PDF:", link["href"])


# Download and extract text
pdf_url = "http://cdnbbsr.s3waas.gov.in/s3ec0490f1f4972d133619a60c30f3559e/documents/misc/practice.pdf_0.pdf"
pdf_text = extract_text_from_pdf(pdf_url)

# Convert to JSON
pdf_data = {"url": pdf_url, "content": pdf_text}
json_output = json.dumps(pdf_data, indent=4)

# Save to file
with open("legal_documents.json", "w") as json_file:
    json_file.write(json_output)

print("PDF data saved to `legal_documents.json`")

