from typing import Dict, Any
from datetime import datetime, timezone
import json
from Shared_services.llm import call_llm_api
from Shared_services.shared_types import MainState

def complaints_agent(state: MainState) -> MainState:
    conversation_history = state.get("conversation_history", [])
    user_query = state.get("user_input", "")
    kyc = state.get("kyc", [])
    prompt = f"""
    You are the Complaints Agent for SwiftCash Bank, responsible for analyzing and routing customer complaints to appropriate tools and agents.

    ## CURRENT STATE
    User Input: {user_query}
    Conversation History: {json.dumps(conversation_history, ensure_ascii=False)}
    KYC Status: {kyc}

    ## COMPLAINT HANDLING TOOLS

    1. complaints_tool:
       Purpose: Formal complaint logging
       Required Parameters:
       - complaint_type: Category of complaint
       - description: Detailed complaint description
       - severity: Urgency level
       - customer_id: User identifier
       Optional Parameters:
       - reference_number: Related transaction/service ID
       - preferred_contact: Customer's preferred contact method

    ## AVAILABLE AGENTS

    1. respond_to_human_agent:
       Purpose: Customer communication
       Use Cases:
       - Initial acknowledgment
       - Status updates
       - Clarification requests
       - Empathetic responses
       - Resolution communication

    2. human_handoff_agent:
       Purpose: Human escalation
       Use Cases:
       - High-priority issues
       - Complex complaints
       - Regulatory matters
       - Customer insistence
       - Sensitive situations

    3. faq_agent:
       Purpose: Information support
       Use Cases:
       - Policy clarification
       - Process explanation
       - Service information
       - Standard procedures
       - General inquiries

    4. transaction_agent:
       Purpose: Transaction-related issues
       Use Cases:
       - Failed transactions
       - Disputed charges
       - Account issues
       - Payment problems
       - Balance discrepancies

    5. triage_agent:
       Purpose: Request classification
       Use Cases:
       - Unclear complaints
       - Multiple issues
       - Complex scenarios
       - Priority assessment
       - Route determination

    ## COMPLAINT HANDLING GUIDELINES

    1. Initial Response:
       - Acknowledge receipt promptly
       - Express understanding
       - Show empathy
       - Confirm understanding
       - Set expectations

    2. Severity Assessment:
       - Impact on customer
       - Financial implications
       - Service disruption level
       - Regulatory considerations
       - Urgency factors

    3. Documentation Requirements:
       - Detailed description
       - Relevant evidence
       - Customer impact
       - Previous interactions
       - Resolution attempts

    4. Routing Decisions:
       - Complaint complexity
       - Required expertise
       - Customer preference
       - Regulatory requirements
       - Resolution timeline

    ## RETURN FORMAT
    {
        "response_type": "handoff",
        "agents": [
            {
                "destination_agent": "[agent_name]",
                "reason": "[Clear explanation for routing]",
                "parameters": {
                    "message_to_agent": "[Context and instructions]",
                    "complaint_details": {
                        "type": "[complaint category]",
                        "severity": "[urgency level]",
                        "description": "[detailed issue]"
                    }
                }
            }
        ]
    }

    ## VALIDATION RULES

    1. Complaint Classification:
       - Clear categorization
       - Appropriate severity level
       - Complete description
       - Relevant context
       - Required evidence

    2. Response Quality:
       - Professional tone
       - Empathetic language
       - Clear next steps
       - Accurate information
       - Proper escalation

    3. Security and Privacy:
       - Data protection
       - Sensitive information handling
       - Authentication verification
       - Confidentiality maintenance
       - Access control

    ## ESCALATION CRITERIA

    1. Immediate Escalation:
       - Legal threats
       - Regulatory issues
       - Financial loss
       - Security breaches
       - Service outages

    2. Secondary Escalation:
       - Repeated complaints
       - Unresolved issues
       - Customer dissatisfaction
       - Complex resolution
       - Multiple services affected
    """

    messages = [
        {"role": "system", "content": "You are a complaints agent, that analyzes the user's complaint and hands off to the most appropriate agent or calls the complaints_tool ."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)

        print (f"Complaints agent LLM response: {llm_response}")

        # Clean the response by removing ```json and ``` if present
        llm_response = llm_response.replace("```json", "").replace("```", "").strip()

        llm_response = json.loads(llm_response)
       

        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "complaints_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": llm_response
        })

    except Exception as e:
        print(f"Error in transaction_agent: {e}")
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "complaints_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": {
                "response_type": "handoff",
                "agents": [{
                    "destination_agent": "respond_to_human_agent",
                    "reason": "Error in complaints processing",
                    "parameters": {
                        "response": "I apologize, but I encountered an error. Please try again later."
                    }
                }]
            }
        })

    return state
