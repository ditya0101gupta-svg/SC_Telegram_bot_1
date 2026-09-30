import os
import logging
import threading
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from langgraph.types import Command
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Simple health server for Render
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")

    def log_message(self, format, *args):
        pass


def run_health_server():
    port = int(os.environ.get("PORT", "10000"))

    logger.info("Starting health server on 0.0.0.0:%s", port)

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    logger.info("Health server started on port %s", port)

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
    
    from agent import agent

    logger.info("Calling agent for incoming Telegram message")

    result = await asyncio.to_thread(
        agent.invoke,
        {"messages": [{"role": "user", "content": update.message.text}]},
        config={"configurable": {"thread_id": str(update.effective_chat.id)}}
    )

    if "__interrupt__" in result:
        await update.message.reply_text(
            "⚠️ Human approval required.\n\n"
            "Reply APPROVE to continue or REJECT to cancel."
        )
        context.user_data["hitl_pending"] = True
        return

        # Handle rate-limit tool errors directly
    for message in result["messages"]:
        if getattr(message, "type", "") == "tool":
            content = getattr(message, "content", "")

            if "temporarily rate-limited" in content.lower():
                await update.message.reply_text(content)
                return

    last_message = result["messages"][-1]

    if getattr(last_message, "content", "") == "STOP_AGENT":
        await update.message.reply_text(
            "🛑 Agent stopped by middleware."
        )
        return

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


    await update.message.reply_text(response_text)

async def handle_approval(update, context):
    decision = update.message.text.strip().lower()

    if not context.user_data.get("hitl_pending"):
        return

    if decision not in {"approve", "reject"}:
        await update.message.reply_text(
            "Please reply with APPROVE or REJECT."
        )
        return

    from agent import agent

    thread_id = str(update.effective_chat.id)

    if decision == "approve":
        resume_command = Command(
            resume={"decisions": [{"type": "approve"}]}
        )
    else:
        resume_command = Command(
            resume={
                "decisions": [{
                    "type": "reject",
                    "message": "User rejected this action."
                }]
            }
        )

    result = await asyncio.to_thread(
        agent.invoke,
        resume_command,
        config={"configurable": {"thread_id": thread_id}}
    )

    context.user_data["hitl_pending"] = False

    response_text = ""

    for message in reversed(result["messages"]):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")

            if isinstance(content, str) and content.strip():
                response_text = content.strip()
                break

    if not response_text:
        response_text = "The action was processed."

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
    MessageHandler(
        filters.Regex(r"(?i)^(approve|reject)$"),
        handle_approval
    )
)

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

app.run_polling()