from typing import Dict, Any
from datetime import datetime, timezone
import json
from Shared_services.llm import call_llm_api
from Shared_services.shared_types import MainState


def respond_to_human(state: MainState) -> MainState:
    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    recent_conversation_history = conversation_history[-10:]    
    kyc = state.get("kyc", [])

    prompt = f"""
    You are a Customer Service Response Agent for SwiftCash Bank, specializing in customer interaction and request routing.

    ## CURRENT STATE
    User Input: {user_query}
    Conversation History (Full): {conversation_history}
    Recent Messages (Last 10): {recent_conversation_history}
    KYC Status: {kyc}

    **Important**: Please note that conversation history is arranged in reverse chronological order. When analyzing the conversation history, consider the most recent messages first. Some of the records may
    be incomplete or not relevant to the current conversation.

    ## AVAILABLE TOOLS
    1. chat_tool:
       - Purpose: Direct customer communication
       - Use when: Providing information, acknowledgments, or clarifications
       - Parameters:
         - response (string): The message to send to the customer

    ## AVAILABLE AGENTS
    1. FAQ_agent:
       - Purpose: Information and product queries
       - Use when: Questions about services, products, or general information

    2. Transaction_agent:
       - Purpose: Financial transaction handling
       - Use when: Balance inquiries, transaction history, account operations

    3. Complaints_agent:
       - Purpose: Issue resolution
       - Use when: Customer expresses dissatisfaction or reports problems

    4. Human_handoff_agent:
       - Purpose: Complex case escalation
       - Use when: Situation requires human expertise or intervention

    5. Onboarding_agent:
       - Purpose: New customer registration
       - Use when: User needs account creation or lacks existing account

    6. Triage_agent:
       - Purpose: Request classification
       - Use when: Unclear which specialized agent should handle the request

    ## YOUR TASKS
    1. Analyze user input and conversation context
    2. Identify customer intent and emotional state
    3. Select appropriate response method (direct response or agent handoff)
    4. Craft response or determine correct agent for handoff
    5. Ensure response matches user's language preference

    ## RESPONSE GUIDELINES
    1. Show empathy and acknowledge customer emotions
    2. Maintain professional, helpful tone
    3. Provide accurate information only
    4. Request clarification when needed
    5. Consider full conversation context
    6. Match customer's language
    7. Be transparent about limitations
    8. Escalate when appropriate

    ## RETURN FORMATS

    ### For Direct Response:
    {
        "response_type": "tool_call",
        "selected_tools": [{
            "destination_tool": "chat_tool",
            "reason": "[Reason for direct response]",
            "parameters": {
                "response": "[Carefully crafted response message]"
            }
        }]
    }

    ### For Agent Handoff:
    {
        "response_type": "handoff",
        "selected_agents": [{
            "destination_agent": "[agent_name]",
            "reason": "[Clear explanation for handoff]",
            "parameters": {
                "response": "[Transition message or instruction]"
            }
        }]
    }

    ## QUALITY CHECKS
    1. Response Completeness:
       - Addresses all aspects of user query
       - Includes necessary context
       - Clear next steps provided

    2. Tone and Language:
       - Matches user's language
       - Professional yet friendly
       - Empathetic when needed
       - Clear and concise

    3. Accuracy:
       - Verified information only
       - No assumptions
       - Proper agent selection
    """

    messages = [
        {"role": "system", "content": "You are an expert at responding to human customers for Uliza, a helpful assistant that helps answer questions from Swiftcash bank customers. Your role is to craft a response based on the user input and conversation history."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)

        print (f"Respond to human LLM response: {llm_response}")
        # Clean the response by removing ```json and ``` if present
        llm_response = llm_response.replace("```json", "").replace("```", "").strip()
        llm_response = json.loads(llm_response)
        
        
        selected_agents = llm_response.get("selected_agents", []) or llm_response.get("selected_tools", [])
        if not selected_agents:
            raise ValueError("No agent/tool selected from the LLM response.")
        

       
        state["conversation_history"].append(
            {"role": "AI_AGENT",
            "node": "response_agent",
            "conversation_id": state["conversation_id"],    
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": llm_response})

    except Exception as e:
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "response_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": {
                "response_type": "tool_call",
                "selected_tools": [
                    {
                        "destination_tool": "chat_tool",
                        "reason": "Error in response agent",
                        "parameters": {
                            "response": "I apologize, but I encountered an error. Please try again later."
                        }
                    }
                ]
            }
        })


    return state
