from typing import List, TypedDict
from langgraph.graph import MemorySaver

class Message(TypedDict):
    role: str   # "user" | "assistant" | "system"
    content: str

class MessagesWithContext(TypedDict):
    messages: List[Message]
    context_chunks: List[str]          # retrieved text
    answer_draft: str | None
    verified: bool | None              # True / False when verified

# Create a MemorySaver instance you can plug into the graph
persistent_memory = MemorySaver()
