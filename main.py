from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
# Agents
from agents.triage_agent import triage_agent
from agents.transaction_agent import transaction_agent
from agents.parameter_collector_agent import parameter_collector_agent
from agents.tool_executor_agent import tool_executor_agent
from agents.tool_response_handler import tool_response_handler
# Shared Services
from shared_services.shared_types import MainState
from shared_services.save_conversation import save_conversation
from shared_services.get_conversation_history import get_conversation_history
from shared_services.logger_setup import setup_logger

logger = setup_logger()


def test_agents():
    # Initialize state first
    state = {
        "user_id": "123464",
        "session_id": "3400000334",
        "user_input": "My account number is 1234567890",
        "conversation_id": "",
        "kyc": {},
        "previous_conversation_history": [],
        "current_conversation_history": [],
        "node_history": [],
        "selected_tools": [],
        "documents": [],
        "tavily_results": [],
        "final_response": "",
        "comprehensive_query": "",
        "similar_questions": []
    }
    state["conversation_id"] = state["user_id"] + state["session_id"] + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    try:
        # Get conversation history
        history = get_conversation_history(
            user_id=state.get("user_id", ""),
            session_id=state.get("session_id", ""),
            conversation_id=state.get("conversation_id", ""),
            limit=10
        )
        
        # Assign conversations to state's conversation_history
        if history["status"] == "success" and history["conversations"]:
            state["previous_conversation_history"] = history["conversations"]
            # Log only the extracted conversation history
            logger.info("Extracted previous conversation history:")
            #logger.info(json.dumps(history["conversations"], indent=2))
        else:
            state["previous_conversation_history"] = None
            logger.info("No previous conversation history found")

        # Run triage agent
        try:
            #result = triage_agent(state)
            #result = transaction_agent(state)
            result = parameter_collector_agent(state)
            result = tool_executor_agent(state)
            result = tool_response_handler(state)
            with open('output.json', 'w') as f:
                json.dump(result, f, indent=4)
            save_conversation(result)

        except Exception as e:
            print(f"Error: {e}")

    except Exception as e:
        print(f"Error in test_agents: {e}")

if __name__ == "__main__":
    test_agents()