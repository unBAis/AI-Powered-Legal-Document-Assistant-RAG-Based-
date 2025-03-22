import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class HybridRetriever:
    def __init__(self, embedding_dim=384, faiss_index_path="faiss.index", data_path="data/legal_docs.json"):
        self.embedding_dim = embedding_dim
        self.faiss_index_path = faiss_index_path
        self.data_path = data_path
        self.documents = []
        self.bm25_corpus = []
        self.bm25 = None
        
        # Load data
        self.load_documents()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if os.path.exists(self.faiss_index_path):
            faiss.read_index(self.faiss_index_path)
        else:
            self.build_index()
    
    def load_documents(self):
        """ Load legal documents from JSON file. """
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
                self.bm25_corpus = [doc["text"] for doc in self.documents]
                self.bm25 = BM25Okapi([doc.split() for doc in self.bm25_corpus])
        else:
            print("No data found. Please add legal documents.")
    
    def build_index(self):
        """ Create FAISS index with embeddings. """
        embeddings = [embedding_model.encode(doc["text"]) for doc in self.documents]
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.faiss_index_path)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """ Hybrid retrieval using BM25 + FAISS vector search. """
        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_k = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_results = [(self.documents[i]["text"], bm25_scores[i]) for i in bm25_top_k]
        
        # FAISS retrieval
        query_embedding = embedding_model.encode(query).reshape(1, -1)
        _, faiss_top_k = self.index.search(query_embedding, top_k)
        faiss_results = [(self.documents[i]["text"], _) for i in faiss_top_k[0]]
        
        # Combine results
        combined_results = bm25_results + faiss_results
        combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:top_k]
        
        return combined_results

# Example usage
if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "What is the penalty for breach of contract?"
    results = retriever.retrieve(query)
    for doc, score in results:
        print(f"Score: {score:.4f}\n{doc}\n")
