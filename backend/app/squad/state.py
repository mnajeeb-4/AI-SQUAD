"""
Squad State Definition
Defines the shared state dictionary passed between LangGraph agents.
"""

from typing import TypedDict

class SquadState(TypedDict):
    messages: list          # Chat history (last 10 messages + new message)
    next_agent: str         # The agent chosen by the Supervisor
    tool_output: str        # Raw output from tools (RAG, Weather API)
    final_answer: str       # The final response sent to the user