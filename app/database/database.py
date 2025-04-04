import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

load_dotenv()

# Check if running on Render and use persistent disk if available
RENDER_DATA_DIR = os.getenv('RENDER_DATA_DIR')
if RENDER_DATA_DIR:
    DATABASE_URL = f"sqlite:///{RENDER_DATA_DIR}/edapi.db"
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////data/edapi.db")

print(f"Using database URL: {DATABASE_URL}")

# Configure SQLite for thread safety
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    # Use StaticPool for SQLite to handle concurrent requests
    poolclass=StaticPool if DATABASE_URL.startswith("sqlite") else None
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
