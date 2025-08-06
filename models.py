from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
# from langgraph.checkpoint.postgres import PostgresSaver



path = 'postgresql://postgres:123456@localhost:5433/postgres'
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


class VectoreStore(Base):
    __tablename__ = 'vectore_stores'
    
    id = Column(Integer, primary_key=True, index=True)
    store_name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="vectore_stores")

# Create the threads table (this is new)
Base.metadata.create_all(engine)


# Add this function to help manage the relationships
def get_user_checkpoints(user_id: int, db_session):
    """Get all checkpoints for a specific user"""
    from sqlalchemy import text
    
    query = text("""
        SELECT c.* 
        FROM checkpoints c
        JOIN threads t ON c.thread_id = t.thread_id
        WHERE t.user_id = :user_id
        ORDER BY c.created_at DESC
    """)
    
    result = db_session.execute(query, {"user_id": user_id})
    return result.fetchall()

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




