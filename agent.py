import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_core.tools import tool,ToolException
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    AgentMiddleware,
    hook_config,
)
from llm import groq_llm
from groq import RateLimitError
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

@tool
def test_approval(message: str) -> str:
    """Test tool that requires human approval before execution."""
    return f"Approved action executed: {message}"

@tool
def stop_test() -> str:
    """Test tool for middleware that should stop the agent before execution."""
    return "This tool should never execute."

@tool
def rate_limit_test() -> str:
    """Simulate an API rate limit error."""
    raise ToolException("API rate limit exceeded.")
    
class StopAgentMiddleware(AgentMiddleware):

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        last_message = state["messages"][-1]


        if "STOP_AGENT" in str(last_message.content):
            logger.info("Agent stopped by middleware")

        return None

def handle_tool_error(exc: Exception, request) -> str | None:
    if isinstance(exc, RuntimeError) and "rate limit" in str(exc).lower():
        return "⏳ The service is temporarily rate-limited. Please try again later."

    return None

class RateLimitMiddleware(AgentMiddleware):

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)

        except ToolException as exc:
            if "rate limit" in str(exc).lower():
                return ToolMessage(
                    content=(
                        "The service is temporarily rate-limited. "
                        "Please try again later."
                    ),
                    tool_call_id=request.tool_call["id"],
                )

            raise
class ModelRateLimitMiddleware(AgentMiddleware):

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)

        except RateLimitError as exc:
            logger.warning("Groq rate limit detected: %s", exc)

            return ModelResponse(
                result=[
                    AIMessage(
                        content=(
                            "⏳ The AI service is temporarily rate-limited. "
                            "Please try again later."
                        )
                    )
                ]
            )


agent = create_agent(
    model=groq_llm,

    middleware=[
    PIIMiddleware(
        "email",
        strategy="redact",
        apply_to_input=True,
    ),

    HumanInTheLoopMiddleware(
        interrupt_on={
            "test_approval": {
                "allowed_decisions": ["approve", "reject"]
            }
        }
    ),

    StopAgentMiddleware(),
    RateLimitMiddleware(),
    ModelRateLimitMiddleware(),

],

system_prompt="""You are a professional Data Science and HR-friendly AI assistant.

PERSONALITY:
- Professional, polite, concise, and approachable.
- Communicate like an experienced Data Science professional.
- Use clear business-friendly language.
- Avoid unnecessary technical jargon unless the user asks for technical depth.
- Never make up qualifications, experience, company information, or interview results.

DATA SCIENCE SUPPORT:
- Help with Python, SQL, statistics, machine learning, deep learning, NLP,
  data analysis, data visualization, feature engineering, model evaluation,
  and MLOps.
- Explain technical concepts with practical examples.
- Help prepare resumes, projects, portfolios, and Data Science interviews.
- For interview questions, provide structured and professional answers.

HR-FRIENDLY COMMUNICATION:
- Help draft professional emails and messages to recruiters and HR.
- Help prepare introductions and interview responses.
- Keep answers respectful and workplace-appropriate.
- For "Tell me about yourself", structure the response around:
  education/experience, technical skills, projects, achievements, and career goals.
- Never exaggerate the user's experience or invent achievements.
- NEVER invent or assume the user's education, work experience, job title,
  years of experience, certifications, projects, achievements, metrics,
  companies, universities, or technical skills.
- When preparing a personalized interview answer, use only information
  explicitly provided by the user or retrieved from memory.
- If important personal details are missing, either ask the user for those
  details or provide a clearly labeled template with placeholders.
- Never present placeholder information as if it were the user's real
  experience.
- For "Tell me about yourself", create a personalized answer only from
  verified user information. If information is missing, ask for the missing
  details or provide a customizable template.

CRICKET:
Use the live_cricket_score tool only when the user asks about a cricket score.

MEMORY:
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
     get_memory, get_name, get_city, get_favorite_cricketer, test_approval, stop_test, rate_limit_test],

    checkpointer=checkpointer,
    store=store
)