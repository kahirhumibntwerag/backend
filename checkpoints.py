from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from database import get_db
from auth import get_current_active_user
from models import User as DBUser, Thread, Checkpoint
import asyncio

router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])

def normalize_state_messages(msgs: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in msgs or []:
        # Support dict-like and LangChain BaseMessage objects
        if isinstance(m, dict):
            typ = m.get("type") or m.get("role")
            content = m.get("content", "")
        else:
            typ = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__.lower()
            content = getattr(m, "content", "")

        # Extract text
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                else:
                    t = getattr(item, "text", None)
                    if t:
                        parts.append(str(t))
            text = "\n".join(parts) if parts else str(content)
        elif isinstance(content, dict):
            text = content.get("text") or str(content)
        else:
            text = str(content)

        sender = "user" if str(typ) in ("human", "user") else ("tool" if str(typ) == "tool" else "model")
        out.append({"sender": sender, "text": text})
    return out

@router.get("/latest")
def latest_checkpoints(
    checkpoint_ns: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    # Get all thread_ids owned by the current user
    thread_ids = [tid for (tid,) in db.query(Thread.thread_id).filter(Thread.user_id == current_user.id).all()]
    if not thread_ids:
        return []

    results: List[Dict[str, Any]] = []
    for tid in thread_ids:
        q = db.query(Checkpoint).filter(Checkpoint.thread_id == tid)
        if checkpoint_ns:
            q = q.filter(Checkpoint.checkpoint_ns == checkpoint_ns)
        cp = q.order_by(Checkpoint.checkpoint_id.desc()).first()
        if cp:
            results.append({
                "thread_id": cp.thread_id,
                "checkpoint_ns": cp.checkpoint_ns,
                "checkpoint_id": cp.checkpoint_id,
                "parent_checkpoint_id": cp.parent_checkpoint_id,
                "type": cp.type,
                "checkpoint": cp.checkpoint,
                "metadata": cp.metadata_,  # mapped from DB column "metadata"
            })
    return results


@router.get("/{thread_id}/latest")
def latest_checkpoint_for_thread(
    thread_id: str,
    checkpoint_ns: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_user),
) -> Optional[Dict[str, Any]]:
    # Ensure the thread belongs to the user (no join)
    owned = db.query(Thread.id).filter(
        Thread.user_id == current_user.id,
        Thread.thread_id == thread_id
    ).first()
    if not owned:
        raise HTTPException(status_code=404, detail="Thread not found for this user")

    q = db.query(Checkpoint).filter(Checkpoint.thread_id == thread_id)
    if checkpoint_ns:
        q = q.filter(Checkpoint.checkpoint_ns == checkpoint_ns)
    cp = q.order_by(Checkpoint.checkpoint_id.desc()).first()
    if not cp:
        return None

    return {
        "thread_id": cp.thread_id,
        "checkpoint_ns": cp.checkpoint_ns,
        "checkpoint_id": cp.checkpoint_id,
        "parent_checkpoint_id": cp.parent_checkpoint_id,
        "type": cp.type,
        "checkpoint": cp.checkpoint,
        "metadata": cp.metadata_,
    }

@router.get("/threads/{thread_id}/messages")
async def thread_messages(
    thread_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_user),
):
    # ensure ownership
    owned = db.query(Thread.id).filter(
        Thread.user_id == current_user.id, Thread.thread_id == thread_id
    ).first()
    if not owned:
        return {"thread_id": thread_id, "messages": []}

    graph = request.app.state.graph
    state = await graph.aget_state({"configurable": {"thread_id": thread_id, "user": current_user.username}})
    msgs = normalize_state_messages(state.values.get("messages", []))
    return {"thread_id": thread_id, "messages": msgs}

@router.get("/messages")
async def all_messages_for_user(
    request: Request,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_user),
):
    tids = [tid for (tid,) in db.query(Thread.thread_id).filter(Thread.user_id == current_user.id).all()]
    if not tids:
        return []
    graph = request.app.state.graph
    states = await asyncio.gather(*[
        graph.aget_state({"configurable": {"thread_id": tid, "user": current_user.username}})
        for tid in tids
    ])
    return [
        {"thread_id": tid, "messages": normalize_state_messages(st.values.get("messages", []))}
        for tid, st in zip(tids, states)
    ]
