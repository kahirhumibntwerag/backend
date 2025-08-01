from fastapi import FastAPI, Query, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from auth import router

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://postgres:123456@localhost:5433/postgres"

search = TavilySearch(max_results=10)
tools = ToolNode(tools=[search])

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and knowledgeable assistant. Always explain concepts clearly and in depth, using practical examples where possible. When responding to user questions, do the following:

Answer the question completely and accurately.

Explain the reasoning or principles behind your answer.

Provide concrete examples or analogies to illustrate your points.

Offer additional tips, alternative solutions, or suggestions that may be helpful.

Tailor your response to the user's level of understanding if detectable. Avoid vague or shallow answers.
always include references to the sources you used to answer the question.
also make the answer long and detailed
include urls to the sources you used to answer the question 
always write everything in markdown format add some colors to the text
the markdown should be neat and nice
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = init_chat_model(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = prompt_template | llm.bind_tools([search])

class State(TypedDict):
    messages: Annotated[list, add_messages]

class ChatRequest(BaseModel):
    message: str
    thread_id: str

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node('tools', tools)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('tools', 'chatbot')
graph_builder.add_conditional_edges('chatbot', tools_condition)

app = FastAPI()

saver_ctx = None
graph = None

@app.on_event("startup")
async def startup():
    global saver, graph, saver_ctx
    saver_ctx = AsyncPostgresSaver.from_conn_string(DB_URI)
    saver = await saver_ctx.__aenter__()
    await saver.setup()
    graph = graph_builder.compile(checkpointer=saver)

@app.on_event("shutdown")
async def shutdown():
    global saver_ctx
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

from langchain_core.messages import AIMessageChunk

@app.get("/chat/stream")
async def chat_stream(thread_id: str, message: str):
    global graph
    if graph is None:
        raise RuntimeError("Graph not initialized")

    async def chat_streamer():
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        ):
            # Filter out non-chat events
            if (
                isinstance(event, dict)
                and event.get("event") == "on_chat_model_stream"
                and "chunk" in event.get("data", {})
            ):
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk):
                    yield f"{chunk.content}"


    return EventSourceResponse(chat_streamer())
