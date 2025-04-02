from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Thread(Base):
    __tablename__ = "threads"

    id = Column(Integer, primary_key=True, index=True)
    ed_thread_id = Column(String, unique=True, index=True)
    title = Column(String)
    content = Column(Text)
    document = Column(Text)
    category = Column(String)
    subcategory = Column(String)
    content_and_img_desc = Column(Text)
    embedding = Column(Text)  # We'll store embeddings as JSON string
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "ed_thread_id": self.ed_thread_id,
            "title": self.title,
            "content": self.content,
            "document": self.document,
            "category": self.category,
            "subcategory": self.subcategory,
            "content_and_img_desc": self.content_and_img_desc,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
