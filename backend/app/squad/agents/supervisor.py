"""
Supervisor Agent
Analyzes user intent and routes the query to the correct specialist agent.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.database import GROQ_API_KEY
from app.squad.state import SquadState

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.0
)

def supervisor_node(state: SquadState):
    """
    Analyzes the user's intent and decides which agent should handle it.
    """
    last_message = state["messages"][-1].content
    
    system_prompt = """You are a Supervisor AI. Your job is to route the user's query to the correct agent.
    Available agents:
    - medical: For health, first aid, or medical questions.
    - legal: For law, legal advice, or Pakistan Code questions.
    - weather: For weather-related questions.
    - general: For greetings, general chat, or anything else.

    Respond ONLY with the agent name (medical, legal, weather, or general). Do not add any other text.
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ])
    
    agent = response.content.strip().lower()
    if agent not in ["medical", "legal", "weather", "general"]:
        agent = "general"
        
    return {"next_agent": agent}