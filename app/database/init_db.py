from app.database.database import engine
from app.models.thread import Base

def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created!")

if __name__ == "__main__":
    init_db()
