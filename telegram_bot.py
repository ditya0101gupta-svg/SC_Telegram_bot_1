import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters

load_dotenv()


# Simple health server for Render
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")

    def log_message(self, format, *args):
        pass


def run_health_server():
    port = int(os.environ.get("PORT", 10000))

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    print(f"Health server running on 0.0.0.0:{port}")
    server.serve_forever()

async def start(update, context):
    await update.message.reply_text(
        "👋 Welcome to Nexora AI!\n\n"
        "I can chat with you, answer questions, and use tools.\n\n"
        "Commands:\n"
        "/help - See what I can do\n"
        "/about - About this bot\n"
        "/clear - Clear conversation"
    )


async def help_command(update, context):
    await update.message.reply_text(
        "🤖 Nexora AI Help\n\n"
        "You can ask me questions and I can use my tools when needed.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show help\n"
        "/about - About Nexora AI\n"
        "/clear - Clear conversation"
    )


async def about(update, context):
    await update.message.reply_text(
        "🚀 Nexora AI\n\n"
        "An AI Telegram assistant built with Python, "
        "LangChain/LangGraph and Groq."
    )


async def clear(update, context):
    from agent import checkpointer

    thread_id = str(update.effective_chat.id)

    checkpointer.delete_thread(thread_id)

    await update.message.reply_text(
        "🧹 Conversation cleared successfully!"
    )

async def reply(update, context):
    print("MESSAGE RECEIVED:", update.message.text)

    from agent import agent

    result = agent.invoke(
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

    response_text = ""

    # Find the latest AI message that actually contains text
    for message in reversed(result["messages"]):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")

            if isinstance(content, str) and content.strip():
                response_text = content.strip()
                break

    # Prevent Telegram's "Message text is empty" error
    if not response_text:
        response_text = "Sorry, I couldn't generate a response. Please try again."

    print("AGENT RESPONSE:", response_text)

    await update.message.reply_text(response_text)

threading.Thread(
    target=run_health_server,
    daemon=True
).start()


app = Application.builder().token(
    os.getenv("TELEGRAM_TOKEN")
).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("about", about))
app.add_handler(CommandHandler("clear", clear))

app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, reply)
)

app.run_polling()