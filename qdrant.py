import os
import shutil
from pathlib import Path
from datetime import datetime
import uuid
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Query, Form
from database import get_db
from sqlalchemy.orm import Session
from models import User as DBUser, VectoreStore
from auth import get_current_active_user

load_dotenv()

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024 # 10GB
ALLOWED_EXTENSIONS = {'.pdf'}
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize embeddings and client
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = QdrantClient("http://localhost:6333")

# Initialize collection
try:
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if "main" not in collection_names:
        client.create_collection(
            collection_name="main", 
            vectors_config={"size": 3072, "distance": "Cosine"}
        )
        print("Created collection: main")
    else:
        print("Collection 'main' already exists")
        
except UnexpectedResponse as e:
    if "already exists" in str(e):
        print("Collection 'main' already exists")
    else:
        print(f"Error with collection creation: {e}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")

# Initialize vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="main",
    embedding=embeddings,
)

# Pydantic models
class CreateVectoreStore(BaseModel):
    store_name: str

class SearchRequest(BaseModel):
    query: str
    store_name: str
    top_k: int = 5

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float

# Utility functions
def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    return Path(filename).name

def cleanup_file(file_path: Path) -> None:
    """Clean up uploaded file."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")

# Router
qdrant_router = APIRouter()

@qdrant_router.post("/create_store")
async def create_store(
    store: CreateVectoreStore, 
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Create a new vector store for the authenticated user."""
    try:
        # Validate store name
        if not store.store_name.strip():
            raise HTTPException(status_code=400, detail="Store name cannot be empty")
        
        # Check if store name already exists for this user
        existing_store = db.query(VectoreStore).filter(
            VectoreStore.store_name == store.store_name,
            VectoreStore.user_id == current_user.id
        ).first()
        
        if existing_store:
            raise HTTPException(
                status_code=409, 
                detail=f"Store '{store.store_name}' already exists for your account"
            )
        
        # Create new store
        new_store = VectoreStore(store_name=store.store_name, user_id=current_user.id)
        db.add(new_store)
        db.commit()
        db.refresh(new_store)
        
        return {
            "message": "Store created successfully",
            "store_id": new_store.id,
            "store_name": new_store.store_name,
            "user_id": new_store.user_id,
            "username": current_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating store: {str(e)}")

@qdrant_router.get("/stores")
async def get_user_stores(
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Get all vector stores for the authenticated user."""
    try:
        stores = db.query(VectoreStore).filter(VectoreStore.user_id == current_user.id).all()
        
        return {
            "username": current_user.username,
            "stores": [
                {
                    "id": store.id,
                    "store_name": store.store_name,
                    "created_at": store.created_at.isoformat() if store.created_at else None
                }
                for store in stores
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stores: {str(e)}")

@qdrant_router.get("/stores/{store_id}")
async def get_store_by_id(
    store_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Get a specific vector store by ID for the authenticated user."""
    try:
        # Get the store and verify ownership
        store = db.query(VectoreStore).filter(
            VectoreStore.id == store_id,
            VectoreStore.user_id == current_user.id
        ).first()
        
        if not store:
            raise HTTPException(
                status_code=404, 
                detail="Store not found or you don't have access to it"
            )
        
        return {
            "id": store.id,
            "store_name": store.store_name,
            "user_id": store.user_id,
            "created_at": store.created_at.isoformat() if store.created_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving store: {str(e)}")

@qdrant_router.post("/add_to_store")
async def add_to_store(
    fileb: Annotated[UploadFile, File()],
    store_name: Annotated[str, Form()],
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Add a PDF file to a vector store."""
    upload_file_path = None
    
    try:
        # Validate file
        validate_file(fileb)
        
        # Verify store exists and user owns it
        store = db.query(VectoreStore).filter(
            VectoreStore.store_name == store_name,
            VectoreStore.user_id == current_user.id
        ).first()
        
        if not store:
            raise HTTPException(status_code=404, detail="Store not found")
        
        # Generate safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = sanitize_filename(fileb.filename)
        upload_filename = f"upload_{timestamp}_{unique_id}_{safe_filename}"
        upload_file_path = UPLOADS_DIR / upload_filename
        
        # Save file
        with open(upload_file_path, "wb") as f:
            content = await fileb.read()
            f.write(content)
        
        # Process PDF
        try:
            loader = PyPDFLoader(str(upload_file_path))
            docs = loader.load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
        
        # Split documents
        text_splitter = SemanticChunker(embeddings)
        all_splits = text_splitter.split_documents(docs)
        
        # Add metadata
        new_splits = []
        for split in all_splits:
            split.metadata.update({
                'store_name': store_name,
                'user': current_user.username,
                'filename': safe_filename,
                'uploaded_at': datetime.now().isoformat()
            })
            new_splits.append(split)
        
        # Add to vector store
        vector_store.add_documents(documents=new_splits)
        
        return {
            "message": "File added to store successfully",
            "filename": safe_filename,
            "chunks_created": len(new_splits),
            "store_name": store_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up uploaded file
        if upload_file_path:
            cleanup_file(upload_file_path)

@qdrant_router.post("/search")
async def search_store(
    search_request: SearchRequest,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Search within a specific store."""
    try:
        # Verify store exists and user owns it
        store = db.query(VectoreStore).filter(
            VectoreStore.store_name == search_request.store_name,
            VectoreStore.user_id == current_user.id
        ).first()
        
        if not store:
            raise HTTPException(status_code=404, detail="Store not found")
        
        # Create Qdrant filter for metadata
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user",
                    match=models.MatchValue(
                        value=current_user.username
                    ),
                ),
                models.FieldCondition(
                    key="metadata.store_name",
                    match=models.MatchValue(
                        value=search_request.store_name
                    ),
                ),
            ]
        )
        
        # Search in vector store with metadata filter
        results = vector_store.similarity_search_with_score(
            query=search_request.query,
            k=search_request.top_k,
            filter=filter,
        )
        
        return {
            "query": search_request.query,
            "store_name": search_request.store_name,
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching store: {str(e)}")

@qdrant_router.delete("/stores/{store_id}")
async def delete_store(
    store_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Delete a vector store by ID for the authenticated user."""
    try:
        store = db.query(VectoreStore).filter(
            VectoreStore.id == store_id,
            VectoreStore.user_id == current_user.id
        ).first()
        
        if not store:
            raise HTTPException(
                status_code=404, 
                detail="Store not found or you don't have access to it"
            )
        
        # TODO: Delete vectors from Qdrant for this store
        # This would require implementing vector deletion by metadata filter
        
        db.delete(store)
        db.commit()
        
        return {
            "message": f"Store '{store.store_name}' deleted successfully",
            "deleted_store_id": store_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting store: {str(e)}")

@qdrant_router.get("/stores/{store_id}/documents")
async def get_store_documents(
    store_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Get all documents in a specific store."""
    try:
        # Verify store exists and user owns it
        store = db.query(VectoreStore).filter(
            VectoreStore.id == store_id,
            VectoreStore.user_id == current_user.id
        ).first()
        
        if not store:
            raise HTTPException(
                status_code=404, 
                detail="Store not found or you don't have access to it"
            )
        
        # Create filter for this store's documents
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user",
                    match=models.MatchValue(
                        value=current_user.username
                    ),
                ),
                models.FieldCondition(
                    key="metadata.store_name",
                    match=models.MatchValue(
                        value=store.store_name
                    ),
                ),
            ]
        )
        
        # Get all documents for this store (limit to reasonable number)
        results = vector_store.similarity_search(
            query="",  # Empty query to get all documents
            k=1000,  # Large number to get all documents
            filter=filter,
        )
        
        # Group by filename
        documents_by_file = {}
        for doc in results:
            filename = doc.metadata.get('filename', 'Unknown')
            if filename not in documents_by_file:
                documents_by_file[filename] = {
                    'filename': filename,
                    'uploaded_at': doc.metadata.get('uploaded_at'),
                    'chunks': []
                }
            documents_by_file[filename]['chunks'].append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return {
            "store_name": store.store_name,
            "store_id": store_id,
            "documents": list(documents_by_file.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")





