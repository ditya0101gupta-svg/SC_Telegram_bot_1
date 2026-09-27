import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from telegram.ext import Application, MessageHandler, filters

from agent import agent

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


async def reply(update, context):
    print("MESSAGE RECEIVED:", update.message.text)

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

    print("AGENT RESPONSE:", result["messages"][-1].content)

    await update.message.reply_text(
        result["messages"][-1].content
    )

    threading.Thread(
    target=run_health_server,
    daemon=True
).start()


app = Application.builder().token(
    os.getenv("TELEGRAM_TOKEN")
).build()

app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, reply)
)

app.run_polling()