from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.cors import CORSMiddleware
from qdrant import vector_store
from qdrant_client import models
from auth import get_current_active_user
from models import User as DBUser
from langchain_core.messages import AIMessageChunk, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
import os
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from qdrant import qdrant_router
from auth import router
from database import get_db
from sqlalchemy.orm import Session
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# ========== LLM + Prompt ==========
llm = init_chat_model(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Here’s a **rewritten and cleaner version** of your prompt, combining both structure and precision — and adding a note about using tools like tables, syntax highlighting, and layout strategies:

---

## ✅ Final Prompt: Structured Markdown Instruction (with Tool Usage)

````text
You are a helpful and knowledgeable assistant. Always respond using **clean, well-formatted GitHub-Flavored Markdown (GFM)**.

Your response will be rendered in a live Markdown interface, so it must be readable, visually structured, and pleasant to read.

---

## 🧱 Markdown Formatting Guidelines

### 📌 General Principles

- Respond **only in Markdown** — no HTML, no escaped characters.
- Keep your answers visually clean, minimal, and well-structured.
- Use spacing between sections, paragraphs, and elements to improve readability.

---

### 🧵 Headings

- Use `#` for the main title, `##` for sub-sections, and `###` for smaller parts.
- After a `#` or `##` heading, insert a horizontal line (`---`) on the next line for visual separation.

**Example:**

```md
## React Context
---
````

---

### 🔠 Text Formatting

* Use `**bold**` for key terms and emphasis.
* Use `_italic_` for soft emphasis or contrast.
* Use `inline code` for referencing code terms inside sentences.

Make sure text formatting doesn’t break sentence flow or spacing.

---

### 🔢 Lists

* Use `-` or `*` for unordered bullet points.
* Use `1.`, `2.` etc. for ordered steps.
* Always leave a blank line before and after lists for spacing.

---

### 💻 Code Blocks

* Use triple backticks (\`\`\`) to wrap multi-line code blocks.
* Always specify the language (like `js`, `python`, `bash`) for syntax highlighting.
* Don't explain what backticks are — just use them.

**Example:**

```python
def greet():
    print("Hello!")
```

---

### 📊 Tables (Use When Comparing)

* Use Markdown tables for clean, visual comparison.
* Always include headers and alignment with `|--|--|`.

**Example:**

```md
| Feature   | Supported |
|-----------|-----------|
| Headings  | ✅        |
| Code      | ✅        |
```

---

### 🔧 Use All Markdown Tools Where Helpful

Apply the full set of Markdown tools where needed:

* `Headings` for hierarchy
* `Lists` for clarity
* `Code blocks` for examples
* `Tables` for comparison
* `Bold/Italic` for emphasis
* `Horizontal lines (---)` to separate sections

Use them naturally to make your response **easy to scan, not just to read**.

---

## 🎯 Your Task

When the user provides a question, respond using all the rules above. Focus on clarity, structure, and Markdown richness. Do not explain the formatting — just use it.

```

---

Would you like a **shorter version** of this for production use? Or should I also prepare a **system message version** for OpenAI tools (like `openai.ChatCompletion.create()`)?
```

     """),
    MessagesPlaceholder(variable_name="messages"),
])

tavily_search = TavilySearch(max_results=5)

# ========== Custom Document Retrieval Tool ==========
@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    """
    Search for relevant documents in the user's personal knowledge base.
    
    Args:
        query: The search query to find relevant documents
    
    Returns:
        A formatted string containing relevant documents and their scores
    """
    # Access configurable parameters correctly
    configurable = config.get("configurable", {})
    user = configurable.get("user")
    store_name = configurable.get("store_name")
    
    # Add validation
    if not user:
        return "Error: User information not available"
    
    if not store_name:
        return "Error: No store name specified. Please provide a store name."
    
    try:
        # Create filter for user's store
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user", 
                    match=models.MatchValue(value=user)
                ),
                models.FieldCondition(
                    key="metadata.store_name", 
                    match=models.MatchValue(value=store_name)
                ),
            ]
        )

        # Search for relevant documents
        try:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=5,
                filter=filter,
            )
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            return f"Error searching document store: {str(e)}"

        if results:
            print(f"Found {len(results)} documents for query: {query}")
            # Format the results
            formatted_docs = "\n\n".join([
                f"**Document {i+1} (Score: {score:.2f}):**\n{doc.page_content}"
                for i, (doc, score) in enumerate(results)
            ])
            
            return f"Found relevant documents from store '{store_name}':\n\n{formatted_docs}"
        else:
            return f"No relevant documents found in store '{store_name}' for the query: '{query}'"
        
    except Exception as e:
        print(f"Error in search_documents: {str(e)}")
        return f"Error searching document store: {str(e)}"

tools = [tavily_search, search_documents]

llm_with_tools = llm.bind_tools(tools)

llm_chain = prompt_template | llm_with_tools

tool_node = ToolNode(tools=tools)



class State(TypedDict):
    messages: Annotated[list, add_messages]



def chatbot(state: dict) -> dict:
    """Generate response using LLM."""
    return {"messages": [llm_chain.invoke(state["messages"])]}

# ========== Graph ==========
graph_builder = StateGraph(State)
graph_builder.add_node('tools', tool_node)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('tools', 'chatbot')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)


# ========== FastAPI App ==========
app = FastAPI()
saver_ctx = None
graph = None

@app.on_event("startup")
async def startup():
    global saver, graph, saver_ctx
    saver_ctx = AsyncPostgresSaver.from_conn_string("postgresql://postgres:123456@localhost:5433/postgres")
    saver = await saver_ctx.__aenter__()
    await saver.setup()
    graph = graph_builder.compile(checkpointer=saver)

@app.on_event("shutdown")
async def shutdown():
    if saver_ctx:
        await saver_ctx.__aexit__(None, None, None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(qdrant_router)

# ========== Manual Token Validation Function ==========
async def validate_token_and_get_user(token: str, db: Session) -> DBUser:
    """Manually validate JWT token and return user."""
    try:
        # Create initial state for token validation
        initial_state = {"token": token, "db": db}
        
        # Use the token validation pipeline from auth.py
        from auth import token_pipeline
        result = token_pipeline.invoke(initial_state)
        
        user = result.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return user
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")
    

# ========== SSE Endpoint ==========
@app.get("/chat/stream")
async def chat_stream(
    thread_id: str,
    message: str,
    token: str,  # Token as URL parameter
    store_name: str = "",  # optional
    db: Session = Depends(get_db)  # Database dependency
):
    if graph is None:
        raise RuntimeError("Graph not initialized")

    # Manually validate token and get user
    current_user = await validate_token_and_get_user(token, db)
    
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    async def streamer():
        # Use proper message object
        initial_state = {"messages": [HumanMessage(content=message)]}
        
        # Add debug logging
        print(f"Starting chat with user: {current_user.username}")
        print(f"Store name: {store_name}")
        print(f"Message: {message}")
        
        async for event in graph.astream_events(
            initial_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "user": current_user.username,
                    "store_name": store_name.strip() or None,
                }
            },
        ):
            # Add debug logging for tool events
            if event.get("event") == "on_tool_start":
                print(f"Tool started: {event.get('data', {}).get('name')}")
            elif event.get("event") == "on_tool_end":
                print(f"Tool ended: {event.get('data', {}).get('name')}")
            
            if (
                isinstance(event, dict)
                and event.get("event") == "on_chat_model_stream"
                and "chunk" in event.get("data", {})
            ):
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk):
                    yield f"{chunk.content}"

    return EventSourceResponse(streamer())





