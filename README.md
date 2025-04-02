# ED API with Telegram Bot

This project provides an API for accessing ED forum threads with a Telegram bot interface.

## Features

- FastAPI backend with SQLAlchemy ORM
- PostgreSQL database for storing threads
- Sentence transformer model for semantic similarity search
- Telegram bot interface for easy access to threads
- Thread similarity search functionality

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a PostgreSQL database

5. Copy `.env.example` to `.env` and fill in your configuration:
```bash
cp .env.example .env
```

6. Create a Telegram bot:
- Message @BotFather on Telegram
- Use the `/newbot` command
- Follow the instructions and get your bot token
- Add the token to your `.env` file

7. Populate the database:
```bash
python -m app.database.populate_db
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.api.main:app --reload
```

2. In a separate terminal, start the Telegram bot:
```bash
python -m app.bot.bot
```

## Usage

The Telegram bot supports the following commands:
- `/start` - Start the bot
- `/help` - Show help message
- `/search <query>` - Search for threads
- `/similar <thread_id>` - Find similar threads

## API Endpoints

- `GET /threads/` - List threads (with pagination)
- `GET /threads/{thread_id}` - Get a specific thread
- `GET /search/similar/{thread_id}` - Find similar threads

## Development

The project structure is organized as follows:
```
app/
├── api/
│   └── main.py
├── bot/
│   └── bot.py
├── database/
│   ├── database.py
│   └── populate_db.py
└── models/
    └── thread.py
```
