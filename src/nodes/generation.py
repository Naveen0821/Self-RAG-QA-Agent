from langgraph.graph import Node
from src.llm.model_loader import get_llm
from src.state.message_state import MessagesWithContext
from pathlib import Path

GEN_PROMPT = Path("src/prompts/generation_prompt.txt").read_text()

class GenerationNode(Node):
    def __init__(self):
        self.llm = get_llm()

    def __call__(self, state: MessagesWithContext):
        context = "\n\n".join(state["context_chunks"])
        user_q = state["messages"][-1]["content"]
        prompt = GEN_PROMPT.format(context=context, question=user_q)
        answer = self.llm([{"role": "user", "content": prompt}]).content
        state["answer_draft"] = answer
        return state
