from langgraph.graph import StateGraph, END
from typing import Dict, Any, Annotated
from agents.triage_agent import triage
from agents.transaction_agent import transaction_agent
from agents.complaints_agent import complaints_agent
from agents.parameter_collector_agent import parameter_collector_agent
from agents.human_handoff_agent import human_handoff_agent
from agents.tool_executor_agent import tool_executor_agent

def should_end(state: Dict[str, Any]) -> bool:
    """Determine if conversation should end"""
    # Add logic to determine if conversation should end
    return False

def get_next_agent(state: Dict[str, Any]) -> str:
    """
    Determine next agent based on conversation state
    """
    # Get latest AI response
    conversation_history = state.get("conversation_history", [])
    latest_response = next(
        (msg["content"] for msg in reversed(conversation_history)
         if msg["role"] == "AI_AGENT" and "content" in msg),
        None
    )

    if not latest_response:
        return "triage"

    # Extract destination from response
    if "agents" in latest_response:
        agent = latest_response["agents"][0]["destination_agent"]
        return agent.lower().replace("_agent", "")
    
    return "end"

def create_conversation_graph() -> StateGraph:
    """
    Create the conversation flow graph
    """
    # Create graph
    workflow = StateGraph(name="conversation_flow")

    # Add nodes
    workflow.add_node("triage", triage)
    workflow.add_node("transaction", transaction_agent)
    workflow.add_node("complaints", complaints_agent)
    workflow.add_node("parameter_collector", parameter_collector_agent)
    workflow.add_node("human_handoff", human_handoff_agent)
    workflow.add_node("tool_executor", tool_executor_agent)

    # Add conditional edges
    workflow.add_conditional_edges(
        "triage",
        get_next_agent,
        {
            "transaction": "transaction",
            "complaints": "complaints", 
            "parameter_collector": "parameter_collector",
            "human_handoff": "human_handoff",
            "tool_executor": "tool_executor",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "transaction",
        get_next_agent,
        {
            "parameter_collector": "parameter_collector",
            "tool_executor": "tool_executor",
            "end": END
        }
    )

    # Add similar edges for other agents...

    # Set entry point
    workflow.set_entry_point("triage")

    return workflow 