from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src import memory
from src.graph import build_graph

app = FastAPI(title="Solar/RES Agentic Assistant API")
_graph = build_graph()
memory.init_db()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    route: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.conversation_id and memory.conversation_exists(req.conversation_id):
        conversation_id = req.conversation_id
    else:
        conversation_id = memory.create_conversation(req.conversation_id)

    history = memory.get_recent_turns(conversation_id, n_turns=memory.DEFAULT_MEMORY_WINDOW)
    memory.save_message(conversation_id, "user", req.message)

    result = _graph.invoke(
        {
            "user_input": req.message,
            "history": history,
            "conversation_id": conversation_id,
            "memory_window": memory.DEFAULT_MEMORY_WINDOW,
        }
    )
    response = result["response"]
    memory.save_message(conversation_id, "assistant", response)

    return ChatResponse(response=response, conversation_id=conversation_id, route=result.get("route"))


@app.get("/conversations/{conversation_id}/history")
def get_history(conversation_id: str):
    if not memory.conversation_exists(conversation_id):
        return {"error": "conversation not found"}
    return {"history": memory.get_full_history(conversation_id)}