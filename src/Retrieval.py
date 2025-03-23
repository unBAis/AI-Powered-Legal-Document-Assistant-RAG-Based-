import faiss
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import json
import os

# Load pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load legal documents
def load_legal_documents(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [doc["text"] for doc in data]

# Tokenize for BM25
def preprocess_for_bm25(docs):
    return [doc.lower().split() for doc in docs]

# Create FAISS index
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Generate document embeddings
def compute_embeddings(docs):
    return np.array(embedding_model.encode(docs))

# Hybrid Retrieval (BM25 + FAISS)
def retrieve_legal_docs(query, top_k=3):
    # Load documents
    documents = load_legal_documents("data/legal_docs.json")

    # BM25 Retrieval
    tokenized_docs = preprocess_for_bm25(documents)
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    # FAISS Retrieval
    query_embedding = embedding_model.encode([query])
    faiss_index = build_faiss_index(compute_embeddings(documents))
    _, faiss_top_indices = faiss_index.search(query_embedding, top_k)

    # Combine Results
    retrieved_indices = list(set(bm25_top_indices) | set(faiss_top_indices[0]))
    retrieved_docs = [documents[i] for i in retrieved_indices]

    return retrieved_docs
