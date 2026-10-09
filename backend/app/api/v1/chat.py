"""
Chat API Endpoint
Handles incoming chat requests, loads memory, and invokes the LangGraph Squad.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.squad.graph import squad_graph
from app.core.database import get_recent_messages, save_message

router = APIRouter()

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    agent: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Load Context (Last 10 messages)
        recent_messages = get_recent_messages(request.thread_id, limit=10)
        
        # 2. Format messages for LangGraph
        formatted_messages = []
        for msg in recent_messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(AIMessage(content=msg["content"]))
                
        # Add the new user message
        formatted_messages.append(HumanMessage(content=request.message))
        
        # 3. Save user message to DB
        save_message(request.thread_id, "user", request.message)
        
        # 4. Invoke the LangGraph Squad
        initial_state = {
            "messages": formatted_messages,
            "next_agent": "",
            "tool_output": "",
            "final_answer": ""
        }
        
        # Run the graph
        result = squad_graph.invoke(initial_state)
        
        final_answer = result.get("final_answer", "I'm sorry, I couldn't process that.")
        routed_agent = result.get("next_agent", "general")
        
        # 5. Save assistant message to DB
        save_message(request.thread_id, "assistant", final_answer, agent=routed_agent)
        
        return ChatResponse(reply=final_answer, agent=routed_agent)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")