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
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

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
    help_text = """
                Available commands:

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
        response = requests.post(
            f"{API_URL}/search/input",
            json={"text": query}
        )
        similar_threads = response.json()

        if not similar_threads:
            update.message.reply_text('No similar questions found.')
            return

        message = f'🔍 Found {len(similar_threads)} relevant questions:\n\n'
        for thread in similar_threads:
            similarity = thread.get('similarity', 0) * 100
            message += display_thread(thread, similarity)


        update.message.reply_text(message)

    except Exception as e:
        update.message.reply_text(f'Error occurred: {str(e)}')





def similar(update: Update, context: CallbackContext) -> None:
    """Find similar threads."""
    if not context.args:
        update.message.reply_text('Please provide a thread ID after /similar')
        return

    thread_id = context.args[0]
    try:
        response = requests.get(f"{API_URL}/search/similar/{thread_id}")
        similar_threads = response.json()

        if not similar_threads:
            update.message.reply_text('No similar threads found.')
            return

        message = "🔍 Similar threads:\n\n"
        for thread in similar_threads:
            similarity = thread.get('similarity', 0) * 100
            message += display_thread(thread, similarity)
        update.message.reply_text(message)

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
            response = requests.get(f"{API_URL}/search/similar/{thread_id}")
            similar_threads = response.json()

            if not similar_threads:
                query.message.reply_text('No similar threads found.')
                return

            message = "🔍 Similar threads:\n\n"
            for thread in similar_threads:
                similarity = thread.get('similarity', 0) * 100
                message += display_thread(thread)

            query.message.reply_text(message)

        except Exception as e:
            query.message.reply_text(f'Error occurred: {str(e)}')

def main() -> None:
    """Start the bot."""
    updater = Updater(TELEGRAM_TOKEN)

    dispatcher = updater.dispatcher

    # Basic commands
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # Search commands
    dispatcher.add_handler(CommandHandler("similar", similar))
    dispatcher.add_handler(CommandHandler("find", search_questions))
    # Analysis commands

    # Button handler
    dispatcher.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
