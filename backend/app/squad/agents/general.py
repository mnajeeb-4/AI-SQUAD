"""
General Agent
Handles greetings, general chat, and fallback queries.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from app.core.database import GROQ_API_KEY
from app.squad.state import SquadState

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.7
)

def general_node(state: SquadState):
    system_prompt = "You are a helpful AI assistant. Respond politely to the user."
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    
    content = response.content
    if not content:
        content = "Hello! How can I help you today?"
        
    return {"final_answer": content}