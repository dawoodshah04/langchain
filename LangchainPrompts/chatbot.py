from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

# ── Model ────────────────────────────────────────────────────────────────────
# ChatHuggingFace wraps HuggingFaceEndpoint and uses the 'conversational'
# task — required by the Groq provider (not 'text-generation').
_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=_endpoint)


# ── Dynamic Prompt Template ──────────────────────────────────────────────────
# {role}    → customisable AI persona (e.g. "Python tutor", "travel guide")
# {topic}   → optional topic focus injected into the system instruction
# {history} → full conversation memory passed via MessagesPlaceholder
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a {role}. "
     "You specialize in {topic}. "
     "Always be concise, helpful, and stay in your assigned role."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}"),
])

chain = prompt_template | llm


# ── File-based Persistent Memory ─────────────────────────────────────────────
HISTORY_DIR = "chat_histories"

def get_file_history(session_id: str) -> FileChatMessageHistory:
    """
    Return a FileChatMessageHistory backed by a JSON file.

    File path: chat_histories/<session_id>.json
    The directory is created automatically if it doesn't exist.
    Every add_user_message() / add_ai_message() call writes to disk
    immediately — no manual save step needed.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path)


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_history(messages: list) -> str:
    """Pretty-print the conversation history with role labels."""
    if not messages:
        return "  (no history yet)"
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"  🧑 Human : {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"  🤖 AI    : {msg.content}")
        elif isinstance(msg, SystemMessage):
            lines.append(f"  ⚙️  System: {msg.content}")
    return "\n".join(lines)


def chat_once(user_input: str,
              file_history: FileChatMessageHistory,
              role: str = "helpful assistant",
              topic: str = "general knowledge") -> str:
    """
    Send a single message and get an AI reply.
    Automatically saves both turns to the JSON file on disk.

    Args:
        user_input   : The human's message.
        file_history : FileChatMessageHistory (reads & writes a JSON file).
        role         : AI persona injected into the system prompt.
        topic        : Subject the AI specialises in.

    Returns:
        ai_reply (str)
    """
    # .messages property reads all past messages from the JSON file on disk
    response = chain.invoke({
        "role": role,
        "topic": topic,
        "history": file_history.messages,   # ← loaded from disk
        "user_input": user_input,
    })

    ai_reply = response.content if hasattr(response, "content") else str(response)

    # Persist both turns to disk immediately
    file_history.add_user_message(user_input)  # ← writes to JSON file
    file_history.add_ai_message(ai_reply)       # ← writes to JSON file

    return ai_reply


# ── Interactive Chat Loop ─────────────────────────────────────────────────────
def run_chat():
    print("=" * 60)
    print("  LangChain Chatbot  |  type 'quit' or 'exit' to stop")
    print("=" * 60)

    # Each unique session_id gets its own JSON history file
    session_id = input("Enter session name (e.g. 'work', 'study'): ").strip() \
                 or "default"

    role  = input("Enter AI role  (default: helpful assistant): ").strip() \
            or "helpful assistant"
    topic = input("Enter AI topic (default: general knowledge) : ").strip() \
            or "general knowledge"

    # Load (or create) the file-backed history for this session
    file_history = get_file_history(session_id)

    past_count = len(file_history.messages)
    if past_count:
        print(f"\n📂 Resumed session '{session_id}' "
              f"— {past_count} previous messages loaded.\n")
        print("── Previous History ──────────────────────────────────")
        print(format_history(file_history.messages))
        print("──────────────────────────────────────────────────────\n")
    else:
        print(f"\n✅  New session '{session_id}' started.")

    print(f"🤖 AI is a '{role}' specialising in '{topic}'.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(f"👋 Goodbye! History saved to "
                  f"'{HISTORY_DIR}/{session_id}.json'")
            break

        ai_reply = chat_once(
            user_input=user_input,
            file_history=file_history,
            role=role,
            topic=topic,
        )

        print(f"🤖 AI : {ai_reply}\n")


# ── Single-message demo (non-interactive) ────────────────────────────────────
def demo_single_message():
    """Quick, one-shot call — saves to chat_histories/demo.json."""
    file_history = get_file_history("demo")
    reply = chat_once(
        user_input="What is LangChain and why should I use it?",
        file_history=file_history,
        role="AI/ML educator",
        topic="LangChain and LLM frameworks",
    )
    print("── Single Message Demo ───────────────────────────────")
    print(format_history(file_history.messages))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        # python chatbot.py --demo   → one-shot single message
        demo_single_message()
    else:
        # python chatbot.py          → interactive loop
        run_chat()