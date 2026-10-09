"""
Chat API Endpoint
-----------------
Handles incoming chat requests, loads memory, and invokes the LangGraph Squad.
Supports Dedicated Bot Channels (Section 7.2) via the optional 'target_agent' field.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.squad.graph import squad_graph
from app.core.database import get_recent_messages, save_message, supabase

router = APIRouter()

# Namespace for deterministic thread IDs (per user + per agent)
THREAD_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

VALID_AGENTS = {"medical", "legal", "weather", "general"}


class ChatRequest(BaseModel):
    thread_id: str                          # The user's base ID (used as user_id)
    message: str
    target_agent: Optional[str] = None      # NEW: medical, legal, weather, or general


class ChatResponse(BaseModel):
    reply: str
    agent: str


def get_agent_thread_id(user_id: str, agent: str) -> str:
    """
    Generate a deterministic thread ID for a given (user, agent) pair.
    This ensures each dedicated bot channel has its own conversation.
    """
    return str(uuid.uuid5(THREAD_NAMESPACE, f"{user_id}:{agent}"))


def ensure_thread_exists(thread_id: str, user_id: str, title: str = "My Chat") -> None:
    """
    Create the thread row in Supabase if it doesn't already exist.
    """
    existing = supabase.table("threads").select("id").eq("id", thread_id).execute()
    if not existing.data:
        supabase.table("threads").insert({
            "id": thread_id,
            "user_id": user_id,
            "title": title,
        }).execute()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_id = request.thread_id
        target_agent = (request.target_agent or "").strip().lower()

        # 1. Determine which thread to use based on mode
        if target_agent and target_agent in VALID_AGENTS:
            # Direct mode: separate thread per agent
            actual_thread_id = get_agent_thread_id(user_id, target_agent)
            thread_title = f"Chat with {target_agent.title()}"
        else:
            # Auto mode: use the user's main thread
            actual_thread_id = user_id
            thread_title = "Squad Chat"
            target_agent = ""  # ensure it's empty for auto mode

        # 2. Ensure the thread exists
        ensure_thread_exists(actual_thread_id, user_id, thread_title)

        # 3. Load context (last 10 messages from this thread)
        recent_messages = get_recent_messages(actual_thread_id, limit=10)

        # 4. Format messages for LangGraph
        formatted_messages = []
        for msg in recent_messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(AIMessage(content=msg["content"]))

        formatted_messages.append(HumanMessage(content=request.message))

        # 5. Save the user's message to DB
        save_message(actual_thread_id, "user", request.message)

        # 6. Invoke the LangGraph Squad
        initial_state = {
            "messages": formatted_messages,
            "next_agent": target_agent,        # pre-set so we know which agent responded
            "target_agent": target_agent,      # if empty, Supervisor routes
            "tool_output": "",
            "final_answer": "",
        }

        result = squad_graph.invoke(initial_state)

        final_answer = result.get("final_answer", "I'm sorry, I couldn't process that.")
        routed_agent = result.get("next_agent") or target_agent or "general"

        # 7. Save the assistant's message to DB
        save_message(actual_thread_id, "assistant", final_answer, agent=routed_agent)

        return ChatResponse(reply=final_answer, agent=routed_agent)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))