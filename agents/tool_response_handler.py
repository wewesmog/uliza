from typing import Dict, Any
from datetime import datetime, timezone
import json
from shared_services.llm import call_llm_api
from shared_services.shared_types import MainState
from shared_services.extract_and_parse_json import extract_and_parse_json
from shared_services.logger_setup import setup_logger
from tools.transaction_tools import TOOLS

logger = setup_logger()

def tool_response_handler(state: MainState) -> MainState:
    """
    Agent responsible for handling tool responses and determining next actions.
    """
    previous_conversation_history = state.get("previous_conversation_history", [])
    current_conversation_history = state.get("current_conversation_history", [])
    user_query = state.get("user_input", "")
    kyc = state.get("kyc", [])

    
    response_from_tools = [record.get("response_from_tool", "") 
                         for record in current_conversation_history 
                         if record.get("role") == "AI_AGENT"
                         and record.get("node") == "tool_executor_agent" 
                         and record.get("destination_agent") == "tool_response_handler"
                         and record.get("conversation_id") == state.get("conversation_id")
                         ]

    print(f"Response from tool(s) for conversation: {state.get('conversation_id')}: {response_from_tools}")
    
    if not response_from_tools:
        state["current_conversation_history"].append({
            "role": "AI_AGENT",
            "node": "tool_response_handler",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": {
                "response_type": "handoff",
                "selected_agents": [{
                    "destination_agent": "triage_agent",
                    "reason": "No tool response found, returning to triage for further assistance",
                    "parameters": {
                        "context": "Tool execution failed or returned no response",
                        "agent_parameters": {}
                    }
                }]
            }
        })
        return state

    prompt = f"""
    You are SwiftCash Bank's Tool Response Handler Agent, responsible for processing tool responses and determining appropriate next actions.
    Keep in mind the current_state below, and use the Response from tool(s) first, before considering the other information.
    ## CURRENT STATE
    Response from tool(s): {response_from_tools}
    User Input: {user_query}
    previous conversation history: {previous_conversation_history}
    current conversation history: {current_conversation_history}
    KYC Status: {kyc}
    
    **Important**: Please note that conversation history is arranged in reverse chronological order. When analyzing the conversation history, consider the most recent messages first. Some of the records may
    be incomplete or not relevant to the current conversation.

    ## AVAILABLE TOOLS

    1. chat_tool:
       Purpose: Communicate with user
       When to use:
       - Transaction confirmations
       - Status updates
       - Additional instructions
       - Error notifications

    ## AVAILABLE AGENTS

    1. parameter_collector_agent:
       Purpose: Collect missing or invalid parameters
       When to call:
       - Need additional transaction details
       - Follow-up transaction parameters
       - Validation requirements

    2. transaction_agent:
       Purpose: Handle financial transactions
       When to call:
       - Additional transactions needed
       - Transaction chain completion
       - Related financial operations

    3. complaints_agent:
       Purpose: Handle transaction issues
       When to call:
       - Partial transaction success
       - User satisfaction check
       - Service quality issues

    4. human_handoff_agent:
       Purpose: Escalate to human agent
       When to call:
       - Complex transaction scenarios
       - Special handling required
       - Security verification needed

    ## RESPONSE EVALUATION RULES

    1. Success Scenarios:
       - Complete success: Inform user and check if additional services needed
       - Partial success: Handle remaining steps
       - Conditional success: Verify conditions and proceed accordingly

    2. Follow-up Actions:
       - Balance check after transfers
       - Transaction verification
       - Related service offerings
       - Additional security measures

    3. User Experience:
       - Clear communication
       - Be creative in your communication
       - Be friendly and engaging
       - Ask the user if they have any other questions or transactions they want to perform
       - Based on the previous transaction, suggest the next best transaction to the user
       - Proactive suggestions
       - Service continuity
       - User satisfaction check

    ## RETURN FORMAT
    Strictly follow the format below, do not add any other fields or information
    
       For tool calls *IMPORTANT* Use this format strictly for tool calls, e.g chat_tool:
    {{
        "response_type": "tool_call",
        "selected_tools": [
            {{
                "destination_tool": "tool_name",  # Use exact tool names - use snake_case naming convention
                "tool_type": "can be transactional or informational",
                "reason": "[Clear explanation for tool use]",
                "parameters": {{
                    "response": "[Message conten to human or tool/agent ]",
                    "next_best_agent": "[agent name for follow-up] - # By default, the next best agent is yourself - tool_executor_agent, unless you are sure another agent is better suited to handle the next step",
                    "context" : "What is currently being handled e.g user is asking for balance, I have requested them for account number" ,
                    "tool_parameters": {{ "#Required and/or available transaction or customer parameters, empty if missing - use exact parameter names e.g account_number, amount"
                        "[parameter_name]": "[value]",
                        "[parameter_name]": "[value]",
                        "[parameter_name]": "[value]",
                        "...": "..."
                    }}
                }}
            }}
        ]
    }}

    For agent handoffs:
    {{
        "response_type": "handoff",
        "selected_agents": [
            {{
                "destination_agent": "[exact_agent_name]",  # Use exact agent names - use snake_case naming convention 
                "reason": "[Clear explanation for handoff]"
                "parameters": {{ 
                    "response": "[Message content to agent]",
                    "context" : "What is currently being handled e.g user is asking for balance, I have requested them for account number",
                    "agent_parameters": {{ "#Required and/or available transaction or customer parameters, empty if missing - use exact parameter names e.g account_number, amount"
                        "[parameter_name]": "[value]",
                        "[parameter_name]": "[value]",
                        "[parameter_name]": "[value]",
                        "...": "..."
                    }}
                }}
            }}
        ]
    }}


    Note: 
    - Analyze the tool response thoroughly
    - Consider both immediate and follow-up actions
    - Make decisions based on transaction context
    - Consider user's potential next needs
    - Maintain transaction chain when needed
    """

    messages = [
        {"role": "system", "content": "You are a tool response handler focusing on processing transaction results and determining appropriate next actions."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        print(f"Tool response handler LLM response: {llm_response}")
        parsed_response = extract_and_parse_json(llm_response)

        state["current_conversation_history"].append({
            "role": "AI_AGENT",
            "node": "tool_response_handler",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": parsed_response
        })

    except Exception as e:
        print(f"Error in tool response handler: {e}")
        state["current_conversation_history"].append({
            "role": "AI_AGENT",
            "node": "tool_response_handler",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": {
                "response_type": "tool_call",
                "selected_tools": [{
                    "destination_tool": "chat_tool",
                    "reason": "Error in handling tool response",
                    "parameters": {
                        "message_to_human": "I apologize, but I encountered an error processing the transaction response. Please try again later.",
                        "next_best_agent": "triage_agent"
                    }
                }]
            }
        })

    return state