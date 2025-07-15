from langgraph.graph import Node
from src.tools.vector_store import get_retriever
from src.state.message_state import MessagesWithContext

class RetrievalNode(Node):
    def __init__(self, vector_store):
        self.retriever = get_retriever(vector_store)

    def __call__(self, state: MessagesWithContext):
        query = state["messages"][-1]["content"]
        docs = self.retriever.get_relevant_documents(query)
        state["context_chunks"] = [d.page_content for d in docs]
        return state
