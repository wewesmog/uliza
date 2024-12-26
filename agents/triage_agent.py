from typing import Dict, Any
from datetime import datetime, timezone
import json
from shared_services.llm import call_llm_api
from shared_services.shared_types import MainState
from shared_services.extract_and_parse_json import extract_and_parse_json
from shared_services.logger_setup import setup_logger
from shared_services.get_conversation_history import get_conversation_history


logger = setup_logger()

def triage_agent(state: MainState) -> MainState:
    #logger.info("Kicked off triage agent with state: %s", 
                   #json.dumps(state, indent=2))


    previous_conversation_history =  state.get("previous_conversation_history", [])
    current_conversation_history =  state.get("current_conversation_history", [])
    user_query = state.get("user_input", "")
    kyc = state.get("kyc", [])

    prompt = f"""
    You are the Triage Agent for SwiftCash Bank, responsible for analyzing customer requests and routing them to the most appropriate specialized agent.
    Keep in mind the current state below:

    ## CURRENT STATE
    User Input: {user_query}
    Previous Conversation History for previous conversations: {previous_conversation_history}
    Current Conversation History for this conversation turn: {current_conversation_history}
    KYC Status: {kyc}

    **Important**: Please note that conversation history is arranged in reverse chronological order. When analyzing the conversation history, consider the most recent messages first. Some of the records may
    be incomplete or not relevant to the current conversation.

    ## AVAILABLE AGENTS

    1. faq_agent:
       Purpose: Information and product queries
       Use Cases:
       - General product information
       - Service explanations
       - Policy questions
       - Account features
       - Interest rates and fees

    2. transaction_agent:
       Purpose: Financial operations
       Use Cases:
       - Balance inquiries
       - Fund transfers
       - Transaction history
       - Bill payments
       - Account operations

    3. complaints_agent:
       Purpose: Issue resolution
       Use Cases:
       - Service dissatisfaction
       - Account access issues
       - Failed transactions
       - Service delays
       - Technical problems

    4. human_handoff_agent:
       Purpose: Complex case escalation
       Use Cases:
       - Explicit human request
       - Complex issues
       - Sensitive matters
       - Regulatory requirements
       - High-value transactions

    ## AVAILABLE TOOLS
    1. chat_tool:
       Purpose: Direct customer interaction. Use this to talk to human
       Use Cases:
       - General inquiries
       - Clarification needs
       - Chitchat handling
       - Status updates
       - Multi-step processes

    ## TRIAGE GUIDELINES

    1. Request Analysis:
       - Identify primary customer intent
       - Consider conversation context
       - Check KYC status relevance
       - Evaluate request complexity
       - Assess urgency level

    2. Agent Selection:
       - Choose based on primary intent
       - Consider secondary requirements
       - Multiple agents if needed
       - Prioritize specialized handling
       - Ensure proper sequencing

    3. Language Handling:
       - Match user's language
       - Consider user's tone 
       - Maintain consistency
       - Consider regional context
       - Preserve meaning accuracy
       - Support multilingual needs

    4. Security and Compliance:
       - Verify KYC requirements
       - Check authorization levels
       - Follow security protocols
       - Maintain data privacy
       - Comply with regulations

  
    ## RETURN FORMAT
    IMPORTANT: You must use EXACTLY this format - no variations allowed:

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
                    "context" : "What is currently being handled e.g user is asking for balance, I have requested them for account number"
                    "tool_parameters": {{ "#Available transaction or customer parameters, empty if missing - use exact parameter names e.g account_number, amount"
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
                    "context" : "What is currently being handled e.g user is asking for balance, I have requested them for account number"
                    "agent_parameters": {{ "#Available transaction or customer parameters, empty if missing - use exact parameter names e.g account_number, amount"
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


    ## DECISION RULES

    1. Priority Routing:
       - Complaints take precedence
       - Security issues escalate
       - Time-sensitive matters prioritized
       - Complex cases to specialists
       - Simple queries to FAQ

    2. Multi-Agent Scenarios:
       - Sequential handling when needed
       - Clear handoff sequence
       - Proper context transfer
       - Avoid redundant routing
       - Maintain conversation flow

    3. Quality Checks:
       - Verify agent availability
       - Confirm routing logic
       - Check parameter completeness
       - Validate handoff context
       - Ensure proper documentation
    """


    messages = [
        {"role": "system", "content": "You are a triage agent for Uliza, a helpful assistant that helps answer questions from Swiftcash bank customers."},
        {"role": "user", "content": prompt}]

    try:
        llm_response = call_llm_api(messages)
        print (f"Triage LLM response: {llm_response}")
        # Clean the response by removing ```json and ``` if present
        parsed_response = extract_and_parse_json(llm_response)


        state["current_conversation_history"].append({
            "role": "user", 
            "conversation_id": state["conversation_id"],    
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": user_query})
        
        
        state["current_conversation_history"].append(
            {"role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],    
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": parsed_response})

    except Exception as e:
        print(f"error: {e}")
        state["current_conversation_history"].append({
            "role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": {
                "response_type": "tool_call",
                "agents": [
                    {
                        "tool": "Respond to Human Tool",
                        "reason": "Error parsing LLM response",
                        "parameters": {
                            "message_for_human": "I apologize, but I encountered an error. Please try again later.",
                            "next_best_agent": "triage agent"
                        }

                    }
                ]
            }
        })

    #log the state
    #logger.info("Completed triage agent with state: %s", 
                   #json.dumps(state, indent=2))
    return state
