import os
from langchain_groq import ChatGroq

def get_llm(model_name=None, temperature=0.2):
    model_name = model_name or os.getenv("GROQ_MODEL", "llama3-8b-8192")
    return ChatGroq(model=model_name, temperature=temperature)
