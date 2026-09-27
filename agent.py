import os
import psycopg
from psycopg.rows import dict_row

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from llm import groq_llm
from tool import live_cricket_score

DB_URI = os.getenv("DATABASE_URL")

conn = psycopg.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(conn)
checkpointer.setup()

store_conn = psycopg.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
)

store = PostgresStore(conn)
store.setup()


@tool
def save_memory(key: str, value: str, config: RunnableConfig) -> str:
    """Save an important fact about the user to long-term memory."""
    user_id = config["configurable"]["thread_id"]
    store.put(("user", user_id), key, {"value": value})
    return f"Saved {key} to memory."

@tool
def get_memory(key: str, config: RunnableConfig) -> str:
    """Retrieve a fact about the user from long-term memory."""
    user_id = config["configurable"]["thread_id"]
    result = store.get(("user", user_id), key)

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
- When the user tells you an important personal fact, save it using save_memory.
- When the user asks about a stored personal fact, ALWAYS call get_memory before answering.
- For the user's name, ALWAYS use the key "name".
- If get_memory returns a value, use that value in your answer.
- Do not claim that a memory is missing until you have called get_memory.
- Do not save passwords, API keys, tokens, or other secrets.
""",

    tools=[live_cricket_score, save_memory,
     get_memory],

    checkpointer=checkpointer,
    store=store
)