from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    CallbackQueryHandler
)
import requests
import os
import sys
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def check_environment():
    """Check if all required environment variables are set."""
    missing_vars = []

    # Check for Telegram token
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        missing_vars.append("TELEGRAM_TOKEN")
    else:
        print(f"✓ TELEGRAM_TOKEN found (starts with: {TELEGRAM_TOKEN[:4]}...)")

    # Check for API key
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        missing_vars.append("API_KEY")
    else:
        print(f"✓ API_KEY found (starts with: {API_KEY[:4]}...)")

    if missing_vars:
        print("\n❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your environment.")
        print("Required variables:")
        print("TELEGRAM_TOKEN=your_telegram_bot_token")
        print("API_KEY=your_api_key")
        print("\nCurrent environment variables:")
        print("PATH:", os.getenv("PATH", "Not set"))
        print("PYTHONPATH:", os.getenv("PYTHONPATH", "Not set"))
        print("All environment variables:", list(os.environ.keys()))
        sys.exit(1)

    return TELEGRAM_TOKEN, API_KEY

# Get environment variables
TELEGRAM_TOKEN, API_KEY = check_environment()
API_URL = os.getenv("API_URL", "http://localhost:8000")
# Headers for API requests
HEADERS = {"X-API-Key": API_KEY}

def make_api_request(method: str, endpoint: str, **kwargs):
    """Make an authenticated request to the API."""
    url = f"{API_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(
            method,
            url,
            headers=HEADERS,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e.response, 'status_code') and e.response.status_code == 403:
            raise ValueError("Invalid API key or authentication failed")
        raise e

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    welcome_text = """
🤖 Hi! I'm your ED Search Assistant!

I can help you find and analyze ED threads. Here are my main commands:

📝 Basic Search:
/find <your question> - Search for similar questions
/part <project part> - Show questions for a specific project part
2


Try /help for more details!
"""
    update.message.reply_text(welcome_text)

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    help_text = """Available commands:

📝 Search Commands:
/find <question> - Find similar questions
/similar <thread_id> - Find similar threads to a specific one

Example usage:
• /find how to implement the cache?
• /similar 1234567890
"""
    update.message.reply_text(help_text)

def search_questions(update: Update, context: CallbackContext) -> None:
    """Search for questions (replaces old search command)."""
    if not context.args:
        update.message.reply_text('Please provide your question after /find')
        return

    query = ' '.join(context.args)
    try:
        similar_threads = make_api_request(
            'POST',
            '/search/input',
            json={"text": query}
        )

        if not similar_threads:
            update.message.reply_text('No similar questions found.')
            return

        message = f'🔍 Found {len(similar_threads)} relevant questions:\n\n'
        for thread in similar_threads:
            similarity = thread.get('similarity', 0) * 100
            message += display_thread(thread, similarity)

        update.message.reply_text(message)

    except ValueError as e:
        update.message.reply_text(f'Authentication error: {str(e)}')
    except Exception as e:
        update.message.reply_text(f'Error occurred: {str(e)}')

def similar(update: Update, context: CallbackContext) -> None:
    """Find similar threads."""
    if not context.args:
        update.message.reply_text('Please provide a thread ID after /similar')
        return

    thread_id = context.args[0]
    try:
        similar_threads = make_api_request('GET', f'/search/similar/{thread_id}')

        if not similar_threads:
            update.message.reply_text('No similar threads found.')
            return

        message = "🔍 Similar threads:\n\n"
        for thread in similar_threads:
            similarity = thread.get('similarity', 0) * 100
            message += display_thread(thread, similarity)
        update.message.reply_text(message)

    except ValueError as e:
        update.message.reply_text(f'Authentication error: {str(e)}')
    except Exception as e:
        update.message.reply_text(f'Error occurred: {str(e)}')

def display_thread(thread, similarity):
    message =  f"• {thread['title']}\n"
    message += f"• EdWard has found a {similarity:3.0f}%  similarity\n"
    message += f"https://eu.edstem.org/courses/1932/discussion/{thread['ed_thread_id']}\n\n"

    return message

def button(update: Update, context: CallbackContext) -> None:
    """Handle button presses."""
    query = update.callback_query
    query.answer()

    if query.data.startswith('similar_'):
        thread_id = query.data.split('_')[1]
        try:
            similar_threads = make_api_request('GET', f'/search/similar/{thread_id}')

            if not similar_threads:
                query.message.reply_text('No similar threads found.')
                return

            message = "🔍 Similar threads:\n\n"
            for thread in similar_threads:
                similarity = thread.get('similarity', 0) * 100
                message += display_thread(thread, similarity)

            query.message.reply_text(message)

        except ValueError as e:
            query.message.reply_text(f'Authentication error: {str(e)}')
        except Exception as e:
            query.message.reply_text(f'Error occurred: {str(e)}')

def main() -> None:
    """Start the bot."""
    try:
        print("\n🤖 Starting bot initialization...")
        print(f"🌐 API URL: {API_URL}")

        if not TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_TOKEN is empty or not set")

        print(f"🔑 Using token starting with: {TELEGRAM_TOKEN[:4]}...")
        updater = Updater(TELEGRAM_TOKEN)
        dispatcher = updater.dispatcher

        # Basic commands
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))

        # Search commands
        dispatcher.add_handler(CommandHandler("similar", similar))
        dispatcher.add_handler(CommandHandler("find", search_questions))

        # Button handler
        dispatcher.add_handler(CallbackQueryHandler(button))

        print("📡 Starting polling...")
        updater.start_polling()
        print("✅ Bot is ready!")
        updater.idle()

    except Exception as e:
        print(f"\n❌ Error starting bot: {str(e)}")
        print("\nDebug information:")
        print(f"TELEGRAM_TOKEN exists: {'Yes' if TELEGRAM_TOKEN else 'No'}")
        print(f"TELEGRAM_TOKEN length: {len(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else 0}")
        print(f"API_URL: {API_URL}")
        sys.exit(1)

if __name__ == '__main__':
    main()
