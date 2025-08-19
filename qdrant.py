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
import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Query, Form
from fastapi.responses import StreamingResponse
from database import get_db
from sqlalchemy.orm import Session
from models import User as DBUser, VectoreStore, UserFile
from auth import get_current_active_user

load_dotenv()

load_dotenv()

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024 # 10GB
ALLOWED_EXTENSIONS = {'.pdf'}
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize embeddings and client
# Initialize embeddings and client
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
_q_timeout = float(os.getenv("QDRANT_TIMEOUT", "120"))
client = QdrantClient(os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=_q_timeout)

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
        
    # Ensure payload indexes used in filters exist
    try:
        client.create_payload_index("main", field_name="metadata.user", field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    try:
        client.create_payload_index("main", field_name="metadata.filename", field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass

except UnexpectedResponse as e:
    if "already exists" in str(e):
        print("Collection 'main' already exists")
    else:
        print(f"Error with collection creation: {e}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="main",
    embedding=embeddings,
)

# Batched add to avoid timeouts on large uploads
def _add_documents_batched(docs: List[Document], batch_size: int | None = None) -> None:
    if not docs:
        return
    try:
        bs_env = int(os.getenv("QDRANT_BATCH_SIZE", "64"))
    except Exception:
        bs_env = 64
    bs = batch_size or bs_env
    start = 0
    while start < len(docs):
        end = min(start + bs, len(docs))
        batch = docs[start:end]
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                vector_store.add_documents(documents=batch)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        if last_err is not None:
            # Fallback: try smaller sub-batches
            if bs > 8:
                half = max(8, bs // 2)
                for mid in range(start, end, half):
                    sub = docs[mid:min(mid+half, end)]
                    vector_store.add_documents(documents=sub)
            else:
                raise last_err
        start = end

class SearchRequest(BaseModel):
    query: str
    file_names: Optional[List[str]] = None
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

@qdrant_router.post("/files/upload")
async def upload_file(
    fileb: Annotated[UploadFile, File()],
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Upload a file for the authenticated user and index it in Qdrant."""
    upload_file_path = None
    try:
        validate_file(fileb)
        # Generate safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = sanitize_filename(fileb.filename)
        upload_filename = f"upload_{timestamp}_{unique_id}_{safe_filename}"
        upload_file_path = UPLOADS_DIR / str(current_user.id) / upload_filename
        upload_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save file
        with open(upload_file_path, "wb") as f:
            content = await fileb.read()
            f.write(content)

        # Persist file record
        uf = UserFile(
            user_id=current_user.id,
            file_name=safe_filename,
            content_type=fileb.content_type,
            size=len(content),
            path=str(upload_file_path),
        )
        db.add(uf)
        db.commit()
        db.refresh(uf)

        # Load and split
        try:
            if upload_file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(upload_file_path))
                docs = loader.load()
            else:
                docs = [Document(page_content=upload_file_path.read_text(errors="ignore"), metadata={})]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

        text_splitter = SemanticChunker(embeddings)
        all_splits = text_splitter.split_documents(docs)

        now_iso = datetime.now().isoformat()
        enriched = []
        for split in all_splits:
            split.metadata.update({
                'user': current_user.username,
                'filename': safe_filename,
                'uploaded_at': now_iso,
                'file_id': uf.id,
            })
            enriched.append(split)

        _add_documents_batched(enriched)

        return {
            "id": uf.id,
            "file_name": uf.file_name,
            "size": uf.size,
            "content_type": uf.content_type,
            "uploaded_at": uf.uploaded_at.isoformat() if uf.uploaded_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    finally:
        pass

@qdrant_router.get("/files")
async def list_files(
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    try:
        rows = db.query(UserFile).filter(UserFile.user_id == current_user.id).order_by(UserFile.uploaded_at.desc()).all()
        return [{
            "id": r.id,
            "file_name": r.file_name,
            "size": r.size,
            "content_type": r.content_type,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        } for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

@qdrant_router.get("/files/{file_id}")
async def get_file_meta(
    file_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    try:
        uf = db.query(UserFile).filter(UserFile.id == file_id, UserFile.user_id == current_user.id).first()
        if not uf:
            raise HTTPException(status_code=404, detail="File not found")
        return {
            "id": uf.id,
            "file_name": uf.file_name,
            "size": uf.size,
            "content_type": uf.content_type,
            "uploaded_at": uf.uploaded_at.isoformat() if uf.uploaded_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")

@qdrant_router.get("/files/{file_id}/download")
async def download_file(
    file_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    try:
        uf = db.query(UserFile).filter(UserFile.id == file_id, UserFile.user_id == current_user.id).first()
        if not uf:
            raise HTTPException(status_code=404, detail="File not found")
        p = Path(uf.path)
        if not p.exists():
            raise HTTPException(status_code=410, detail="File missing on server")
        return StreamingResponse(open(p, "rb"), media_type=uf.content_type or "application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="{uf.file_name}"'
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@qdrant_router.post("/search")
async def search_files(
    search_request: SearchRequest,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Search within the user's files (optionally restricted to specific file names)."""
    try:
        must = [
            models.FieldCondition(
                key="metadata.user",
                match=models.MatchValue(value=current_user.username)
            ),
        ]
        should = []
        for fn in (search_request.file_names or []):
            should.append(
                models.FieldCondition(
                    key="metadata.filename",
                    match=models.MatchValue(value=fn)
                )
            )
        filter = models.Filter(must=must, should=should or None)

        results = vector_store.similarity_search_with_score(
            query=search_request.query,
            k=search_request.top_k,
            filter=filter,
        )
        
        return {
            "query": search_request.query,
            "file_names": search_request.file_names or [],
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
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@qdrant_router.delete("/files/{file_id}")
async def delete_file(
    file_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    try:
        uf = db.query(UserFile).filter(UserFile.id == file_id, UserFile.user_id == current_user.id).first()
        if not uf:
            raise HTTPException(status_code=404, detail="File not found")

        # Delete vectors for this file (user + file_id)
        client.delete(
            collection_name="main",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.user", match=models.MatchValue(value=current_user.username)),
                        models.FieldCondition(key="metadata.file_id", match=models.MatchValue(value=file_id)),
                    ]
                )
            ),
        )

        # Delete file from disk
        try:
            p = Path(uf.path)
            if p.exists():
                p.unlink()
        except Exception:
            pass

        db.delete(uf)
        db.commit()
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@qdrant_router.patch("/files/{file_id}/rename")
async def rename_file(
    file_id: int,
    new_name: str,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Rename a file and update Qdrant payload to reflect the new filename."""
    try:
        uf = db.query(UserFile).filter(UserFile.id == file_id, UserFile.user_id == current_user.id).first()
        if not uf:
            raise HTTPException(status_code=404, detail="File not found")
        if not new_name or not new_name.strip():
            raise HTTPException(status_code=400, detail="New name is required")
        new_safe = sanitize_filename(new_name)

        src = Path(uf.path)
        dst = src.with_name(new_safe)
        try:
            src.rename(dst)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error renaming file: {str(e)}")

        # Update DB
        uf.file_name = new_safe
        uf.path = str(dst)
        db.add(uf)
        db.commit()
        db.refresh(uf)

        # Update Qdrant payload filename for all points of this file
        try:
            client.set_payload(
                collection_name="main",
                payload={"metadata.filename": new_safe},
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.user", match=models.MatchValue(value=current_user.username)),
                        models.FieldCondition(key="metadata.file_id", match=models.MatchValue(value=file_id)),
                    ]
                ),
            )
        except Exception:
            # Non-fatal; vectors remain searchable by file_id anyway
            pass

        return {
            "id": uf.id,
            "file_name": uf.file_name,
            "size": uf.size,
            "content_type": uf.content_type,
            "uploaded_at": uf.uploaded_at.isoformat() if uf.uploaded_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error renaming file: {str(e)}")

@qdrant_router.put("/files/{file_id}/replace")
async def replace_file(
    file_id: int,
    fileb: Annotated[UploadFile, File()],
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Replace a file's contents and reindex Qdrant vectors for this file."""
    try:
        uf = db.query(UserFile).filter(UserFile.id == file_id, UserFile.user_id == current_user.id).first()
        if not uf:
            raise HTTPException(status_code=404, detail="File not found")

        # Delete existing vectors for this file
        client.delete(
            collection_name="main",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.user", match=models.MatchValue(value=current_user.username)),
                        models.FieldCondition(key="metadata.file_id", match=models.MatchValue(value=file_id)),
                    ]
                )
            ),
        )

        # Replace file on disk (keep same file name)
        dest = Path(uf.path)
        try:
            content = await fileb.read()
            dest.write_bytes(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error replacing file contents: {str(e)}")

        # Update DB metadata
        uf.size = len(content)
        uf.content_type = fileb.content_type
        db.add(uf)
        db.commit()
        db.refresh(uf)

        # Reindex
        try:
            if dest.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(dest))
                docs = loader.load()
            else:
                docs = [Document(page_content=dest.read_text(errors="ignore"), metadata={})]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

        text_splitter = SemanticChunker(embeddings)
        all_splits = text_splitter.split_documents(docs)
        now_iso = datetime.now().isoformat()
        enriched = []
        for split in all_splits:
            split.metadata.update({
                'user': current_user.username,
                'filename': uf.file_name,
                'uploaded_at': now_iso,
                'file_id': uf.id,
            })
            enriched.append(split)
        _add_documents_batched(enriched)

        return {"id": uf.id, "file_name": uf.file_name, "reindexed_chunks": len(enriched)}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error replacing file: {str(e)}")
@qdrant_router.get("/files/{file_id}/chunks")
async def get_file_chunks(
    file_id: int,
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
    db: Session = Depends(get_db)
):
    """Return all indexed chunks for a given file (for inspection)."""
    try:
        uf = db.query(UserFile).filter(UserFile.id == file_id, UserFile.user_id == current_user.id).first()
        if not uf:
            raise HTTPException(status_code=404, detail="File not found")

        filter = models.Filter(
            must=[
                models.FieldCondition(key="metadata.user", match=models.MatchValue(value=current_user.username)),
                models.FieldCondition(key="metadata.file_id", match=models.MatchValue(value=file_id)),
            ]
        )
        results = vector_store.similarity_search(
            query="context",
            k=1000,
            filter=filter,
        )
        documents_by_file = {
            uf.file_name: {
                'filename': uf.file_name,
                'uploaded_at': uf.uploaded_at.isoformat() if uf.uploaded_at else None,
                'chunks': [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in results
                ]
            }
        }
        return {
            "file_id": uf.id,
            "file_name": uf.file_name,
            "documents": list(documents_by_file.values())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")





