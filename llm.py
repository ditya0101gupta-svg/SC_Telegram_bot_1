import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

#Groq
groq_llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq"
)