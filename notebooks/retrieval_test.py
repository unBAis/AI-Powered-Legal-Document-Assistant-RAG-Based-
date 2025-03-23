

import requests
from bs4 import BeautifulSoup

url = "https://main.sci.gov.in/judgments"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract judgment links
for link in soup.find_all("a", href=True):
    if "pdf" in link["href"]:
        print("Found judgment PDF:", link["href"])
