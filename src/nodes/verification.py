from langgraph.graph import Node
from src.llm.model_loader import get_llm
from pathlib import Path
from src.state.message_state import MessagesWithContext

VER_PROMPT = Path("src/prompts/verification_prompt.txt").read_text()

class VerificationNode(Node):
    def __init__(self):
        self.llm = get_llm()

    def __call__(self, state: MessagesWithContext):
        answer = state["answer_draft"]
        context = "\n\n".join(state["context_chunks"])
        prompt = VER_PROMPT.format(answer=answer, context=context)
        verdict = self.llm([{"role": "user", "content": prompt}]).content.strip().lower()
        state["verified"] = "yes" in verdict
        return state
