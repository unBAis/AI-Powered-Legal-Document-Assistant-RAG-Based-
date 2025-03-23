import openai
import os
from Retrieval import HybridRetriever
from typing import List

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def query_llm(prompt: str, model: str = "gpt-4-turbo") -> str:
    """ Query OpenAI's GPT model and return the response. """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a legal expert assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_response(query: str, top_k: int = 5) -> str:
    """ Retrieve relevant documents and generate LLM-based response. """
    retriever = HybridRetriever()
    retrieved_docs = retriever.retrieve(query, top_k)
    
    # Prepare context from retrieved documents
    context = "\n\n".join([doc[0] for doc in retrieved_docs])
    final_prompt = (f"Given the following legal context, answer the query:\n\n"
                    f"{context}\n\n"
                    f"Query: {query}\n"
                    f"Answer in a professional yet simple manner.")
    
    response = query_llm(final_prompt)
    return response

# Example usage
if __name__ == "__main__":
    user_query = "What are the rights of a tenant in case of eviction?"
    print("\nGenerated Response:\n", generate_response(user_query))
