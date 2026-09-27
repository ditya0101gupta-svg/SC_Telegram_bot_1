from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from llm import groq_llm
from tool import live_cricket_score

checkpointer = InMemorySaver()
store = InMemoryStore()

@tool
def save_memory(key: str, value: str) -> str:
    """Save an important fact about the user to long-term memory."""
    store.put(("user",), key, {"value": value})
    return f"Saved {key} to memory."


@tool
def get_memory(key: str) -> str:
    """Retrieve a fact about the user from long-term memory."""
    result = store.get(("user",), key)

    if result is None:
        return f"No memory found for {key}."

    return result.value["value"]

agent = create_agent(
    model=groq_llm,

    system_prompt="""You are a helpful assistant.

For every user question, first call the live_cricket_score tool
using country1="India" and country2="Pakistan".

After receiving the tool result, answer the user's question normally.

LONG-TERM MEMORY:
- When the user tells you an important personal fact, such as their name, save it using the save_memory tool.
- When the user asks about a fact that may be stored in memory, use the get_memory tool.
- For the user's name, use the key "name".
- Do not save passwords, API keys, tokens, or other secrets.
""",

    tools=[live_cricket_score, save_memory,
     get_memory],

    checkpointer=checkpointer,
    store=store
)