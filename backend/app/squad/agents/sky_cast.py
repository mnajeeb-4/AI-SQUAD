"""
Sky Cast (Weather Agent)
Handles weather queries using the Hybrid Weather System (Cache + API).
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.database import GROQ_API_KEY
from app.squad.state import SquadState
from app.squad.tools.weather import get_weather_from_cache, get_weather_from_api

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.0
)

def weather_node(state: SquadState):
    query = state["messages"][-1].content
    
    # Extract city name using LLM (simple extraction)
    extract_prompt = "Extract only the city name from this query. If no city is mentioned, reply 'unknown'."
    response = llm.invoke([SystemMessage(content=extract_prompt), HumanMessage(content=query)])
    city = response.content.strip()
    
    if city.lower() == "unknown" or not city:
        return {"final_answer": "Please specify a city for the weather report."}
    
    # Hybrid Weather System: Check cache first, then API
    weather_data = get_weather_from_cache(city)
    if not weather_data:
        weather_data = get_weather_from_api(city)
        
    return {"final_answer": weather_data}