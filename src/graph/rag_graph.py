from langgraph.graph import StateGraph
from src.nodes.retrieval_node import RetrievalNode
from src.nodes.generation_node import GenerationNode
from src.nodes.verification_node import VerificationNode
from src.state.message_state import MessagesWithContext, persistent_memory

def build_graph(vector_store):
    sg = StateGraph(MessagesWithContext)
    sg.add_node("retrieve", RetrievalNode(vector_store))
    sg.add_node("generate", GenerationNode())
    sg.add_node("verify", VerificationNode())

    # Edges
    sg.add_edge("retrieve", "generate")
    sg.add_edge("generate", "verify")
    # End state is after verification
    sg.set_entry_node("retrieve")
    return sg.compile(memory=persistent_memory)
