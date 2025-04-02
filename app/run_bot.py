import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the Telegram bot."""
    # Load environment variables from scraper/.env
    scraper_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scraper', '.env')
    load_dotenv(scraper_env_path)

    # Check if Telegram token is set
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    if not telegram_token:
        logger.error("TELEGRAM_TOKEN environment variable is not set.")
        logger.error("Please set it in scraper/.env or the main .env file.")
        return 1

    logger.info("Starting Telegram bot...")
    from app.bot.bot import main as run_bot
    run_bot()

    return 0

if __name__ == "__main__":
    exit(main())
