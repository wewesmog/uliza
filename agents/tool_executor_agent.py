from typing import Dict, Any
from datetime import datetime, timezone
import json
from shared_services.llm import call_llm_api
from shared_services.shared_types import MainState
from shared_services.extract_and_parse_json import extract_and_parse_json
from shared_services.logger_setup import setup_logger
from tools.transaction_tools import TOOLS

logger = setup_logger()

def tool_executor_agent(state: MainState) -> MainState:
    """
    Agent responsible for final validation and execution of tool requests.
    """
    previous_conversation_history = state.get("previous_conversation_history", [])
    current_conversation_history = state.get("current_conversation_history", [])
    user_query = state.get("user_input", "")
    kyc = state.get("kyc", [])
   
    prompt = f"""
    You are SwiftCash Bank's Tool Executor Agent, responsible for the final validation and secure execution of financial transactions.

    ## CURRENT STATE
    User Input: {user_query}
    Previous Conversation History for previous conversations: {previous_conversation_history}
    Current Conversation History for this conversation turn: {current_conversation_history}
    KYC Status: {kyc}
   
    **Important**: Please note that conversation history is arranged in reverse chronological order. When analyzing the conversation history, consider the most recent messages first. Some of the records may
    be incomplete or not relevant to the current conversation.

    ## AVAILABLE AGENTS

    1. parameter_collector_agent:
       Purpose: Collect missing or invalid parameters
       When to call:
       - Parameters are missing
       - Validation fails
       - Need additional information
       - Parameter format issues
       - Incomplete transaction data

    2. transaction_agent:
       Purpose: Handle financial transactions
       When to call:
       - Need transaction rerouting
       - Complex transaction scenarios
       - Multiple transaction types
       - Special handling required

    3. triage_agent:
       Purpose: Re-evaluate user intent
       When to call:
       - User intent unclear
       - Request needs reclassification
       - Multiple requests detected
       - Changed conversation direction

    4. faq_agent:
       Purpose: Handle information requests
       When to call:
       - Questions about process
       - Tool information needed
       - Parameter explanations
       - Policy clarifications

    5. complaints_agent:
       Purpose: Handle user dissatisfaction
       When to call:
       - Transaction failures
       - User frustration
       - Service issues
       - Technical problems

    6. human_handoff_agent:
       Purpose: Escalate to human agent
       When to call:
       - Complex issues
       - High-value transactions
       - Security concerns
       - Multiple failures

    ## AVAILABLE TOOLS AND VALIDATION RULES

    1. account_transfer_tool:
       Purpose : Transfer funds between accounts
       Tool Type : Transactional
       Required Parameters:
       - account_number_from: 8-12 digits, numeric only
       - account_number_to: 8-12 digits, numeric only
       - amount: Numeric, > 0, max 2 decimals
       Optional:
       - reason: Text
       - confirm: Boolean (default: true)

    2. balance_inquiry_tool:
       Purpose : Check account balance
       Tool Type : Transactional
       Required Parameters:
       - account_number: 8-12 digits, numeric only
       Optional:
       - reason: Text
       - confirm: Boolean (default: true)

    3. transaction_history_tool:
       Purpose : View transaction history
       Tool Type : Transactional
       Required Parameters:
       - account_number: 8-12 digits, numeric only
       Optional:
       - date_range: Valid date format
       - transaction_type: Valid type
       - confirm: Boolean (default: true)

    4. paybill_tool:
       Purpose : Pay bills
       Tool Type : Transactional
       Required Parameters:
       - biller_code: Valid format
       - account_number: 8-12 digits
       - amount: Numeric, > 0
       Optional:
       - reason: Text
       - confirm: Boolean (default: true)
    5. chat_tool:
       Purpose : Chat with the user/customer
       Tool Type : Informational
       Required Parameters:
       - None
       Optional:
       - None
    ## RESPONSE FORMATS

    ## RETURN FORMAT
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
                    "context" : "What is currently being handled e.g user is asking for balance, I have requested them for account number",
                    "next_best_agent": "[agent name for follow-up] - # This is the agent that is best suited to handle the next step after tool execution. If unsure, choose yourself - tool_executor_agent",
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
    - Output the JSON only, no extra wordings or characters
    - Use EXACT field names as shown
    - Don't modify the structure. Only respond in JSON
    - Your response must be enclosed in curly brackets and all text in double quotes
    - Don't add extra fields or any information e.g 'Here is the JSON response:'
    - Don't use variations of field names

    ## VALIDATION GUIDELINES
    1. Parameter Completeness:
       - Check all required parameters exist
       - Verify parameter types match requirements
       - Validate parameter formats

    2. Business Rules:
       - Verify transaction limits
       - Check account restrictions
       - Validate against KYC rules

    3. Security Checks:
       - Flag unusual amounts
       - Check for suspicious patterns
       - Verify user permissions

    4. Confirmation Rules:
       - Always require confirmation for:
         * High-value transactions
         * First-time recipients
         * Unusual patterns
         * New biller payments

   
    """

    messages = [
        {"role": "system", "content": "You are a secure financial transaction executor focusing on parameter validation and security compliance. Do not call transactional tools directly, your role is to validate parameters and handoff to the tool_executor_agent"},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        print(f"Tool executor LLM response: {llm_response}")
        parsed_response = extract_and_parse_json(llm_response)

        state["current_conversation_history"].append({
            "role": "AI_AGENT",
            "node": "tool_executor_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": parsed_response
        })

        # Execute tool if response type is tool_call
        if parsed_response.get("response_type") == "tool_call":
            for tool in parsed_response.get("selected_tools", []):
                if tool.get("tool_type") == "transactional":
                    tool_name = tool.get("destination_tool")
                    if tool_name:
                        tool_function = TOOLS.get(tool_name)
                        if tool_function:
                            parameters = tool.get("parameters", {}).get("tool_parameters", {})
                            print(f"Executing tool: {tool_name} with parameters: {parameters}")
                            response = tool_function(**parameters)
                            print(f"Tool executed successfully: {tool_name}")
                            #call the chat tool and send the response
                            state["current_conversation_history"].append({
                                    "role": "AI_AGENT",
                                    "node": "tool_executor_agent",
                                    "response_type": "handoff",
                                    "destination_agent": "tool_response_handler",
                                    "conversation_id": state["conversation_id"],
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "response_from_tool": response,
                                    
                                })
                        else:
                            print(f"Tool not found: {tool_name}")
                            
                            
    except Exception as e:
        print(f"Error in tool executor: {e}")
        state["current_conversation_history"].append({
            "role": "AI_AGENT",
            "node": "tool_executor_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": {
                "response_type": "tool_call",
                "selected_tools": [{
                    "destination_tool": "chat_tool",
                    "reason": "Error in tool execution",
                    "parameters": {
                        "response": "I apologize, but I encountered an error. Please try again later.",
                        "next_best_agent": "triage_agent"
                    }
                }]
            }
        })

    return state