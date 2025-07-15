import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END

from src.tools.pdf_loader import load_pdf_chunks
from src.tools.vector_store import build_vector_store
from src.graph.rag_graph import build_graph
from src.state.message_state import MessagesWithContext

# Load env vars
load_dotenv()

st.set_page_config(page_title="📄 Self-RAG QA Agent", layout="wide")
st.title("📄 Self-RAG QA Agent with LangGraph + Groq")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "graph_app" not in st.session_state:
    st.session_state.graph_app = None

# Upload section
uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
if uploaded_file:
    with st.spinner("📚 Reading and chunking PDF..."):
        file_path = f"/tmp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        chunks = load_pdf_chunks(file_path)
        vs = build_vector_store(chunks)
        st.session_state.vector_store = vs
        st.session_state.graph_app = build_graph(vs)

        st.success("✅ PDF processed and vector store created.")

# Question section
if st.session_state.graph_app:
    st.subheader("💬 Ask a question about the uploaded paper")
    user_query = st.text_input("Your question")

    if st.button("Ask"):
        if not user_query.strip():
            st.warning("⚠ Please enter a question.")
        else:
            with st.spinner("🤖 Retrieving and answering..."):
                messages = [{"role": "user", "content": user_query}]
                state = {
                    "messages": messages,
                    "context_chunks": [],
                    "answer_draft": None,
                    "verified": None
                }
                result = st.session_state.graph_app.invoke(state)

                # Display output
                st.markdown("### ✅ Answer")
                st.write(result["answer_draft"])
                st.markdown(f"**Verification Status:** `{result['verified']}`")

                if result["verified"] is False:
                    st.info("⚠️ This answer may not be fully supported by the source document.")
