import os
from dotenv import load_dotenv
from telegram.ext import Application, MessageHandler, filters

from agent import agent

load_dotenv()

async def reply(update, context):
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": update.message.text
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(update.effective_chat.id)
            }
        }
    )

    await update.message.reply_text(
        result["messages"][-1].content
    )

app = Application.builder().token(
    os.getenv("TELEGRAM_TOKEN")
).build()

app.add_handler(
    MessageHandler(filters.TEXT, reply)
)

app.run_polling()