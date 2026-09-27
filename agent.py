from langchain.agents import create_agent
from langgraph.checkpointer.memory import InMemorySaver
from llm import groq_llm
from tool import live_cricket_score

agent = create_agent(
    model=groq_llm,
    system_prompt="""You are a helpful assistant.

For every user question, first call the live_cricket_score tool
using country1="India" and country2="Pakistan".

After receiving the tool result, answer the user's question normally.
""",
    tools=[live_cricket_score],
    checkpointer=InMemorySaver()
)