"""
Squad Graph Assembly
Connects the Supervisor and all specialist agents into a LangGraph workflow.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from app.squad.state import SquadState
from app.squad.agents.supervisor import supervisor_node
from app.squad.agents.dr_bot import medical_node
from app.squad.agents.legal_eagle import legal_node
from app.squad.agents.sky_cast import weather_node
from app.squad.agents.general import general_node


# --- Conditional Edge Router ---
def route_to_agent(state: SquadState) -> Literal["medical", "legal", "weather", "general"]:
    """
    Reads the 'next_agent' field from the state and routes accordingly.
    """
    return state["next_agent"]


# --- Build the Graph ---
workflow = StateGraph(SquadState)

# Add nodes (agents)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("medical", medical_node)
workflow.add_node("legal", legal_node)
workflow.add_node("weather", weather_node)
workflow.add_node("general", general_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Add conditional edges from supervisor to agents
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "medical": "medical",
        "legal": "legal",
        "weather": "weather",
        "general": "general"
    }
)

# Direct Handoff: All agents go straight to END (Supervisor does NOT summarize)
workflow.add_edge("medical", END)
workflow.add_edge("legal", END)
workflow.add_edge("weather", END)
workflow.add_edge("general", END)

# Compile the graph
squad_graph = workflow.compile()