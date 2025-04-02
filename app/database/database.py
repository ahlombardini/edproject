import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Check if running on Render and use persistent disk if available
RENDER_DATA_DIR = os.getenv('RENDER_DATA_DIR')
if RENDER_DATA_DIR:
    DATABASE_URL = f"sqlite:///{RENDER_DATA_DIR}/edapi.db"
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./edapi.db")

print(f"Using database URL: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
