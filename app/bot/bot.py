from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters, CallbackQueryHandler
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! I can help you search through ED threads. Try /help to see what I can do.')

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
    Available commands:
    /start - Start the bot
    /help - Show this help message
    /search <query> - Search for threads
    /similar <thread_id> - Find similar threads
    /find <text> - Find threads similar to your input text
    """
    update.message.reply_text(help_text)

def search(update: Update, context: CallbackContext) -> None:
    """Search for threads based on query."""
    if not context.args:
        update.message.reply_text('Please provide a search query after /search')
        return

    query = ' '.join(context.args)
    try:
        response = requests.get(f"{API_URL}/threads/", params={"skip": 0, "limit": 5})
        threads = response.json()

        if not threads:
            update.message.reply_text('No threads found.')
            return

        for thread in threads:
            text = f"Title: {thread['title']}\nCategory: {thread['category']}\nID: {thread['ed_thread_id']}"
            keyboard = [[InlineKeyboardButton("Find Similar", callback_data=f"similar_{thread['ed_thread_id']}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            update.message.reply_text(text, reply_markup=reply_markup)

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

        for thread in similar_threads:
            similarity = thread.get('similarity', 0) * 100
            text = f"Title: {thread['title']}\nCategory: {thread['category']}\nSimilarity: {similarity:.2f}%\nID: {thread['ed_thread_id']}"
            update.message.reply_text(text)

    except Exception as e:
        update.message.reply_text(f'Error occurred: {str(e)}')

def find_text(update: Update, context: CallbackContext) -> None:
    """Find threads similar to user input text."""
    if not context.args:
        update.message.reply_text('Please provide some text after /find')
        return

    query_text = ' '.join(context.args)
    try:
        response = requests.post(
            f"{API_URL}/search/input",
            json={"text": query_text}
        )
        similar_threads = response.json()

        if not similar_threads:
            update.message.reply_text('No similar threads found.')
            return

        update.message.reply_text(f'Found {len(similar_threads)} threads similar to: "{query_text}"')

        for thread in similar_threads:
            similarity = thread.get('similarity', 0) * 100
            text = f"Title: {thread['title']}\nCategory: {thread['category']}\nSimilarity: {similarity:.2f}%\nID: {thread['ed_thread_id']}"
            update.message.reply_text(text)

    except Exception as e:
        update.message.reply_text(f'Error occurred: {str(e)}')

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

            for thread in similar_threads:
                similarity = thread.get('similarity', 0) * 100
                text = f"Title: {thread['title']}\nCategory: {thread['category']}\nSimilarity: {similarity:.2f}%\nID: {thread['ed_thread_id']}"
                query.message.reply_text(text)

        except Exception as e:
            query.message.reply_text(f'Error occurred: {str(e)}')

def main() -> None:
    """Start the bot."""
    updater = Updater(TELEGRAM_TOKEN)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("search", search))
    dispatcher.add_handler(CommandHandler("similar", similar))
    dispatcher.add_handler(CommandHandler("find", find_text))
    dispatcher.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
