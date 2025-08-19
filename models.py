from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv
load_dotenv()
import os

path = os.getenv("APP_DATABASE_URL")
print(path)
engine = create_engine(path)
Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    __allow_unmapped__ = True  # Fixed typo
    id = Column(Integer, primary_key=True)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to threads
    threads = relationship("Thread", back_populates="user")
    vectore_stores = relationship("VectoreStore", back_populates="user")

class Thread(Base):
    __tablename__ = 'threads'
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True, nullable=False)  # LangGraph thread_id
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String, nullable=True)  # Optional chat title
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="threads")
    checkpoints = relationship("Checkpoint", back_populates="thread")


class VectoreStore(Base):
    __tablename__ = 'vectore_stores'
    
    id = Column(Integer, primary_key=True, index=True)
    store_name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="vectore_stores")

class UserFile(Base):
    __tablename__ = 'user_files'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True, nullable=False)
    file_name = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    size = Column(Integer, nullable=True)
    path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class Checkpoint(Base):
    __tablename__ = 'checkpoints'

    # Composite PK as per your table
    thread_id = Column(String, ForeignKey('threads.thread_id'), primary_key=True, nullable=False)
    checkpoint_ns = Column(String, primary_key=True, nullable=False)
    checkpoint_id = Column(String, primary_key=True, nullable=False)

    # Other columns you listed
    parent_checkpoint_id = Column(String, nullable=True)
    type = Column(String, nullable=True)
    checkpoint = Column(JSONB, nullable=True)
    metadata_ = Column('metadata', JSONB, nullable=True)  # map to DB column "metadata"

    # Relationship back to Thread
    thread = relationship("Thread", back_populates="checkpoints")

# Create the threads table (this is new)
Base.metadata.create_all(engine)


# Add this function to help manage the relationships
def get_user_checkpoints(user_id: int, db_session):
    return (
        db_session.query(Checkpoint)
        .join(Thread, Thread.thread_id == Checkpoint.thread_id)
        .filter(Thread.user_id == user_id)
        .all()
    )

def create_thread_for_user(user_id: int, thread_id: str, title: str = None, db_session=None):
    """Create a new thread for a user"""
    if db_session is None:
        from database import SessionLocal
        db_session = SessionLocal()
    
    thread = Thread(
        user_id=user_id,
        thread_id=thread_id,
        title=title
    )
    db_session.add(thread)
    db_session.commit()
    return thread




