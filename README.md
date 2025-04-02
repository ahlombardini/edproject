# ED API with Telegram Bot

This project provides an API for accessing ED forum threads with a Telegram bot interface.

## Features

- FastAPI backend with SQLAlchemy ORM
- SQLite database for storing threads
- Sentence transformer model for semantic similarity search
- Telegram bot interface for easy access to threads
- Thread similarity search functionality
- Automatic sync with ED Stem API

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.api.txt
```

4. Copy `.env.example` to `.env` and fill in your configuration:
```bash
cp .env.example .env
```

5. Create a Telegram bot:
- Message @BotFather on Telegram
- Use the `/newbot` command
- Follow the instructions and get your bot token
- Add the token to your `.env` file

## Running the Application

1. Start the FastAPI server:
```bash
python -m uvicorn app.api.main:app --reload
```

2. In a separate terminal, start the Telegram bot:
```bash
python -m app.bot.bot
```

## Deployment on Render

### Setting Up Persistent Storage

1. Create a new Web Service on Render
2. Add a Disk:
   - Go to your service's "Disks" tab
   - Click "Add Disk"
   - Set mount path to `/data`
   - Choose an appropriate size (e.g., 1 GB)

### Environment Variables

Set the following environment variables in your Render dashboard:
- `ED_API_TOKEN`: Your ED Stem API token
- `ED_API_HOST`: ED Stem API host (https://eu.edstem.org/api)
- `ED_COURSE_ID`: Your course ID
- `TELEGRAM_TOKEN`: Your Telegram bot token
- `API_URL`: Your Render service URL
- `SYNC_INTERVAL_MINUTES`: Sync interval (default: 120)
- `MAX_THREADS_PER_SYNC`: Max threads per sync (default: 20)
- `DISABLE_SYNC`: Set to 0 to enable syncing

The application will automatically:
1. Detect it's running on Render
2. Use the persistent disk at `/data` for the database
3. Maintain data between deployments

## API Endpoints

- `GET /threads/` - List all threads
- `GET /threads/{thread_id}` - Get a specific thread
- `GET /threads/search?query=<query>` - Search threads semantically
- `GET /sync/status` - Check sync service status
- `POST /sync/trigger` - Manually trigger a sync
- `GET /sync/threads` - Get thread statistics

## Development

The project structure is organized as follows:
```
app/
├── api/
│   ├── main.py
│   ├── edstem_client.py
│   └── sync_service.py
├── bot/
│   └── bot.py
├── database/
│   └── database.py
└── models/
    └── thread.py
```
