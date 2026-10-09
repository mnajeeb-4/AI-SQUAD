"""
Squad Graph Assembly
Connects the Supervisor and all specialist agents into a LangGraph workflow.
Supports two modes:
  1. Auto mode: Supervisor routes the query.
  2. Direct mode: A specific agent is selected, bypassing the Supervisor.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END

from app.squad.state import SquadState
from app.squad.agents.supervisor import supervisor_node
from app.squad.agents.dr_bot import medical_node
from app.squad.agents.legal_eagle import legal_node
from app.squad.agents.sky_cast import weather_node
from app.squad.agents.general import general_node


# --- Conditional Edge Router (from Supervisor) ---
def route_to_agent(state: SquadState) -> Literal["medical", "legal", "weather", "general"]:
    """
    Reads the 'next_agent' field from the state and routes accordingly.
    """
    return state["next_agent"]


# --- NEW: Conditional Entry Router (bypasses Supervisor if target_agent is set) ---
def route_entry(state: SquadState) -> str:
    """
    Determines the entry point of the graph.
    If a specific target_agent is set, skip the Supervisor and go directly to it.
    Otherwise, start with the Supervisor (auto-routing).
    """
    target = state.get("target_agent", "")
    if target and target in ["medical", "legal", "weather", "general"]:
        return target
    return "supervisor"


# --- Build the Graph ---
workflow = StateGraph(SquadState)

# Add nodes (agents)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("medical", medical_node)
workflow.add_node("legal", legal_node)
workflow.add_node("weather", weather_node)
workflow.add_node("general", general_node)

# NEW: Conditional entry point (Direct mode vs Auto mode)
workflow.add_conditional_edges(
    START,
    route_entry,
    {
        "supervisor": "supervisor",
        "medical": "medical",
        "legal": "legal",
        "weather": "weather",
        "general": "general",
    },
)

# Add conditional edges from supervisor to agents (Auto mode)
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

# Direct Handoff: All agents go straight to END
workflow.add_edge("medical", END)
workflow.add_edge("legal", END)
workflow.add_edge("weather", END)
workflow.add_edge("general", END)

# Compile the graph
squad_graph = workflow.compile()


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