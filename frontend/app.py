import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/query"

# Streamlit UI
st.set_page_config(page_title="Legal AI Assistant", layout="wide")
st.title("📝 Legal AI Assistant - RAG Powered")

st.markdown("🔍 **Ask legal questions, and get AI-powered answers with citations!**")

# User input
query = st.text_area("Enter your legal question:", height=100)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            response = requests.post(API_URL, json={"query": query})

            if response.status_code == 200:
                data = response.json()
                st.subheader("🧠 AI-Generated Answer:")
                st.write(data["answer"])

                st.subheader("📜 Sources & References:")
                for idx, source in enumerate(data["sources"], start=1):
                    st.markdown(f"**{idx}.** {source}")

            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
