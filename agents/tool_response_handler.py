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

    
    response_from_tool = [record.get("response_from_tool", "") 
                         for record in previous_conversation_history 
                         if record.get("role") == "AI_AGENT"
                         and record.get("node") == "tool_executor_agent" 
                         and record.get("destination_agent") == "tool_response_handler"
                         and record.get("conversation_id") == "123456789120241215191717" #state.get("conversation_id")
                         ]

    print(f"Response from tool(s): {response_from_tool}")

    prompt = f"""
    You are SwiftCash Bank's Tool Response Handler Agent, responsible for processing tool responses and determining appropriate next actions.
    Keep in mind the current_state below, and use the Response from tool(s) first, before considering the other information.
    ## CURRENT STATE
    Response from tool(s): {response_from_tool}
    User Input: {user_query}
    Previous Conversation History for previous conversations: {previous_conversation_history}
    Current Conversation History for this conversation turn: {current_conversation_history}
    KYC Status: {kyc}
    

    ## AVAILABLE TOOLS

    1. account_transfer_tool:
       Purpose: Transfer funds between accounts
       When to use:
       - Follow-up transfers needed
       - Batch transfers
       - Related account operations

    2. balance_inquiry_tool:
       Purpose: Check account balance
       When to use:
       - Post-transaction balance check
       - Available funds verification
       - Multiple account checks

    3. transaction_history_tool:
       Purpose: View transaction history
       When to use:
       - Verify transaction completion
       - Check related transactions
       - Transaction pattern analysis

    4. paybill_tool:
       Purpose: Pay bills
       When to use:
       - Additional bill payments
       - Related service payments
       - Scheduled payments

    5. chat_tool:
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
    
       For tool calls:
    {{
        "response_type": "tool_call",
        "selected_tools": [
            {{
                "destination_tool": "tool_name",  # Use exact tool names - use snake_case naming convention
                "tool_type": "can be transactional or informational",
                "reason": "[Clear explanation for tool use]",
                "parameters": {{
                    "response": "[Message conten to human or tool/agent ]",
                    "next_best_agent": "[agent name for follow-up] - # This is the agent that is best suited to handle the next step after tool execution. If unsure, choose yourself - tool_executor_agent",
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