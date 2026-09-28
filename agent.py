import os
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import PIIMiddleware
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

store_pool = ConnectionPool(
    DB_URI,
    min_size=1,
    max_size=5,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    },
)

store = PostgresStore(store_pool)
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

@tool
def get_name(config: RunnableConfig) -> str:
    """Retrieve the user's stored name. Use this when the user asks for their name."""
    user_id = config["configurable"]["thread_id"]

    result = store.get(("user", user_id), "name")

    if result is None:
        return "No name found."

    return result.value["value"]

@tool
def get_city(config: RunnableConfig) -> str:
    """Retrieve the user's stored city. Use this when the user asks where they live."""
    user_id = config["configurable"]["thread_id"]

    result = store.get(("user", user_id), "city")

    if result is None:
        return "No city found."

    return result.value["value"]


@tool
def get_favorite_cricketer(config: RunnableConfig) -> str:
    """Retrieve the user's stored favorite cricketer."""
    user_id = config["configurable"]["thread_id"]

    result = store.get(("user", user_id), "favorite_cricketer")

    if result is None:
        return "No favorite cricketer found."

    return result.value["value"]

agent = create_agent(
    model=groq_llm,

    middleware=[
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
    ],

    system_prompt="""You are a helpful assistant.

Use the live_cricket_score tool only when the user asks about a cricket score.
For memory questions, use the appropriate memory tool directly.
After using a tool, answer the user's question normally.

LONG-TERM MEMORY:
- When the user explicitly tells you a personal fact, ALWAYS save it immediately using save_memory.
- Do not wait for the user to ask you to remember it.
- Save each personal fact separately.
- Use these exact keys when applicable:
  - name
  - city
  - hobby
  - favorite_cricketer
  - favorite_team
  - favorite_food
- For any other personal fact, create a short descriptive key.
- When the user asks about a stored personal fact, ALWAYS retrieve the memory before answering.
- When the user asks for their name, ALWAYS call get_name.
- NEVER use get_memory to retrieve the user's name.
- When the user asks where they live, ALWAYS call get_city.
- NEVER use get_memory to retrieve the user's city.
- When the user asks for their favorite cricketer, ALWAYS call get_favorite_cricketer.
- NEVER use get_memory to retrieve the user's favorite cricketer.
- For other stored personal facts, use get_memory with the correct key.
""",

    tools=[live_cricket_score, save_memory,
     get_memory, get_name, get_city, get_favorite_cricketer],

    checkpointer=checkpointer,
    store=store
)