"""
Legal Eagle (Legal Agent)
Handles legal queries using the Pakistan Law Code RAG.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.database import GROQ_API_KEY
from app.squad.state import SquadState
from app.squad.tools.rag import search_documents

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.2
)

def legal_node(state: SquadState):
    query = state["messages"][-1].content
    context = search_documents(query, match_threshold=0.3, match_count=3)
    
    system_prompt = f"""You are Legal Eagle, a legal assistant specializing in Pakistan Law. Use the following context to answer.
    Context: {context}
    
    IMPORTANT: Make only ONE tool call.
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    content = response.content
    if not content:  # Silence Guard
        content = f"Here is the legal data I found: {context}"
        
    return {"final_answer": content}