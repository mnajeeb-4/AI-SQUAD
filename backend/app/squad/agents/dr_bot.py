"""
Dr. Bot (Medical Agent)
Handles medical and first aid queries using RAG.
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

def medical_node(state: SquadState):
    query = state["messages"][-1].content
    context = search_documents(query, match_threshold=0.3, match_count=3)
    
    system_prompt = f"""You are Dr. Bot, a medical assistant. Use the following First Aid context to answer the user's question.
    Context: {context}
    
    IMPORTANT: Make only ONE tool call. If you don't know the answer, say you don't know.
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    content = response.content
    if not content:  # Silence Guard
        content = f"Here is the medical data I found: {context}"
        
    return {"final_answer": content}