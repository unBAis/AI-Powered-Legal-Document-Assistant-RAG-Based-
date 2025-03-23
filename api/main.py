

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Retrieval import retrieve_legal_docs  # Hybrid search (BM25 + embeddings)
from llm_rag import generate_llm_response  # LLM integration

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str

# Response schema
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
async def query_legal_ai(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve relevant legal documents
    retrieved_docs = retrieve_legal_docs(query)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant legal documents found")

    # Generate LLM response using retrieved docs
    llm_answer = generate_llm_response(query, retrieved_docs)

    return QueryResponse(
        query=query,
        answer=llm_answer,
        sources=retrieved_docs
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
