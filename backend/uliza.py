import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional, TypedDict, Union
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2
import requests
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from psycopg2.extras import Json, RealDictCursor
from google.generativeai import GenerativeModel
import numpy as np
from openai import OpenAI
from tavily import TavilyClient

# Define the main state structure
class MainState(TypedDict):
    user_id: str
    session_id: str
    conversation_id: str
    user_input: str
    selected_tools: List[Dict[str, str]]
    conversation_history: List[Dict[str, str]]
    documents: List[Dict[str, Any]]
    tavily_results: List[Dict[str, Any]]
    final_response: str
    comprehensive_query: str
    similar_questions: List[str]

# Configure APIs
genai.configure(api_key="YOUR_GOOGLE_API_KEY")
client = genai.GenerativeModel('gemini-pro')
openai_client = OpenAI(api_key="")

def call_llm_api(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", max_tokens: int = 2000, temperature: float = 0.3) -> Any:
    """
    Make a call to the OpenAI API for chat completions.
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        raise

def setup_logger(log_level=logging.INFO):
    main_project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    log_folder = os.path.join(main_project_directory, "logs")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file_path = os.path.join(log_folder, f"query_state_log_{datetime.now().strftime('%Y-%m-%d')}.log")
    logger = logging.getLogger("QueryStateLogger")
    if not logger.handlers:
        logger.setLevel(log_level)
        file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=30)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def triage_agent(state: MainState) -> MainState:
    conversation_history = state.get("conversation_history", [])
    user_query = state.get("user_input", "")


    prompt = f"""
    **IMPORTANT**: Please be aware that you are conversing with both human & AI agents. 
    Given the user input, which is what the customer entered, and conversation history which is the conversation between the user and the assistant or from agent to agent, handoff to the appropriate agent(s).

    User input: {user_query}
    Conversation history: {json.dumps(conversation_history, ensure_ascii=False)}
  
  
    Agents available:
    1. **FAQ_agent**: Hand off to this agent when the user is asking questions related to information, services, or products.
    2. **Transactional_agent**: Hand offto this agent when the user is asking questions related to transactions, such as balance inquiries, transaction history, and account details.
    3. **complaints_agent**: Hand off to this agent when the user is complaining or reporting an issue.
    4. **human_handoff_agent**: Hand off to this agent when you need to transfer the conversation to a human agent.
    5. **response_agent**: Hand off to this agent when you want to respond to the customer (human). 

    Guidelines:
    1. Carefully analyze the user input and conversation history and node history to select the appropriate agent(s).
    2. You can handoff to more than one agent if needed.

    3. If the user input is chitchat, or matters not related to banking, or the input is not clear, or it cant be handled by any of the other agents, 
    handoff to the **response_agent** to respond to the user.
    Example:
    User_input: How are you?

    Expected response:
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "response_agent",
                "reason": "EXAMPLE:User query is a chitchat",
                "parameters": {{
                    "response_to_customer": "EXAMPLE: I'm doing great, thank you for asking!"
                }}
            }}
        ]
    }}

    4. If the input is related to specific services, products, or information (including details about loans, banking services, or 
    financial products), handoff to the **FAQ_agent**.
    In addition, remember to create a comprehensive query and a list of similar questions to be used to search for the most relevant information in the documents.
    Example:
    User_input: What is the interest rate on the savings account?

    Expected response:  
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "FAQ_agent",
                "reason":  "EXAMPLE: User query is about specific information or products",
                "parameters": {{
                    "user_query": "{user_query}",
                    "comprehensive_query": "INSTRUCTION: using the user_query and conversation history, create a comprehensive query that will be used to search for the most relevant information in the documents.",
                    "similar_questions": "INSTRUCTION: using the user_query and conversation history, create a list of 3 similar questions that will be used to search for the most relevant information in the documents."
                }}
            }}
        ]
    }}

    5. If the user wants to transact e.g open a new account, check balance, transfer funds, pay bills, handoff to the **Transactional_agent**.

    Example:
    User_input: I want to open a new savings account.

    Expected response:  
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "Transactional_agent",
                "reason":"EXAMPLE: User query is about a transaction",
                "parameters": {{
                    "user_query": "{user_query}"
                }}
            }}
        ]
    }}
    6. For the complaints_agent, handoff to it when:
       - Customer expresses dissatisfaction
       - Reports an issue or problem
       - Disputes a transaction
       - Mentions fraud or security concerns
       - Needs resolution for a service failure

    Example for complaints:
    User_input: I was charged twice for my last transaction and need this resolved.

    Expected response:  
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "complaints_agent",
                "reason":  "EXAMPLE: User is reporting a double charge issue that needs investigation",
                "parameters": {{
                    "complaint_type": "EXAMPLE: transaction_dispute",
                    "user_query": "{user_query}"
                }}
            }}
        ]
    }}
    7. For the human_handoff_agent, handoff to it when:
       - Customer explicitly requests to speak to a human
       - Query is too complex or sensitive for AI handling
       - Multiple failed attempts to resolve an issue
       - Regulatory or compliance-related matters
       - High-value transactions or sensitive account changes

    Example for human handoff:
    User_input: I want to speak with a real person, not a bot.

    Expected response:  
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "human_handoff_agent",
                "reason": "EXAMPLE: Customer explicitly requested human assistance",
                "parameters": {{
                    "user_query": "{user_query}"
                }}
            }}
        ]
    }}
    8. It is possible to handoff to more than one agent.
    Example:
    User_input: what are the loan types available, and what is my account balance?

    Expected response:  
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "FAQ_agent",
                "reason": "EXAMPLE: User query is about specific information or products",
                "parameters": {{
                    "user_query": "{user_query}"
                }}
            }},
            {{
                "destination_agent": "Transactional_agent",
                "reason": "EXAMPLE: User query is about a transaction",
                "parameters": {{
                    "user_query": "{user_query}"
                }}
            }}
        ]
    }}
    9. Users may ask questions in different languages e.g Kiswahili. In such cases, respond in the same language they used.
    10.Ensure that the response is in valid JSON format and matches the structure shown in the examples above.
    
    """

    messages = [
        {
            "role": "system", 
            "content": "You are triage agent, an expert in understanding the customer's intent and handing off to the best agent(s). You are part of a team of agents that make up Uliza, a helpful assistant that helps answer questions from Swiftcash bank customers"
        },
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        llm_response = json.loads(llm_response)
        
       
        
        
        # Update conversation conversation_history
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": llm_response
        })

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response: {e}")
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content":  {{
                            "response_type": "handoff",
                            "agents": [
                                {
                                    "destination_agent": "response_agent",
                                    "reason": "Error parsing LLM response",
                                    "parameters": {{
                                        "response": "I apologize, but I encountered an error. Could you please rephrase your question?"
                                    }}
                                }
                            ]
                            }}})
        
        
    except Exception as e:
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content":  {{
                            "response_type": "handoff",
                            "agents": [
                                {
                                    "destination_agent": "response_agent",
                                    "reason": "Error parsing LLM response",
                                    "parameters": {{
                                        "response": "I apologize, but I encountered an error. Please try again later."
                                    }}
                                }
                            ]
                            }}})

    return state

def transaction_agent(state: MainState) -> MainState:
    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
 

    #TODO: Implement the transaction agent
    prompt = f"""

    **IMPORTANT**: Please be aware that you are conversing with both human & AI agents. 

    You are a transaction agent that helps answer questions related to transactions, such as balance inquiries, transaction history, and account details, as well as perform transactions.
    User query: {user_query}
    Conversation history: {conversation_history}
 

    
    Use User query& conversation history which is the conversation between AI tools & human,  to answer questions and/or perform transactions.
    Ensure you check the conversation history, to determine if the required parameters for the tools are available.

    You are part of a team of agents that make up Uliza, a helpful assistant that helps answer questions from Swiftcash bank customers.
    Your fellow agents are:
    1.  **Transactional_agent**: Yourself!
    2. **FAQ_agent**: Hand off to this agent when the user is asking questions related to information, services, or products; Expected parameters: user_query, comprehensive_query (which is a comprehensive query to be used to search for the most relevant information in the documents), similar_questions (which is a list of 3 similar questions that will be used to search for the most relevant information in the documents)   
    3. **complaints_agent**: Hand off to this agent when the user is complaining or reporting an issue; Expected parameters: complaint_type, user_query
    4. **human_handoff_agent**: Hand off to this agent when you need to transfer the conversation to a human agent; Expected parameters: user_query
    5. **response_agent**: Hand off to this agent when you want to respond to the customer (human); Expected parameters: response_to_customer

    You have access to the following tools, carefully select the appropriate tool(s) to use based on the user query and conversation history:

    1. **Balance_inquiry_tool**: Use this tool to answer questions related to balance inquiries.
        Required parameters:
        - account_number: The account number to check the balance for.
    
    2. **Mini_statement_tool**: Use this tool to answer questions related to mini statements.
        Required parameters:
        - account_number: The account number to check the mini statement for.
        - start_date: The start date of the mini statement.
        - end_date: The end date of the mini statement.
    
    3. **Account_transfer_tool**: Use this tool to perform account transfers.
        Required parameters:
        - account_number: The account number to transfer funds from.
        - destination_account_number: The account number to transfer funds to.
        - amount: The amount to transfer.

    4. **Bill_payment_tool**: Use this tool to perform bill payments.
        Required parameters:
        - account_number: The account number to pay the bill from.
        - biller_name: The name of the biller.
        - amount: The amount to pay the bill.
    
    5. **Account_opening_tool**: Use this tool to perform account openings.
        Required parameters:
        - account_type: The type of account to open.
        - ID_number: The ID number of the customer.
        - phone_number: The phone number of the customer.

    IMPORTANT:
    Give your response in valid JSON format and match the structure shown in the examples. You can either handoff to another agent or call the tool(s) based on the user query and conversation history.
    To handoff to another agent, use the format below:
    {{
        "response_type": "handoff",
        "agents": [
            {{
                "destination_agent": "agent_name",
                "reason": "reason for handoff",
                "parameters": {{
                    
                }}
            }}
        ]
    }}

    To call a tool(s), use the format below:
    {{
        "response_type": "call_tool",
        "tools": [
            {{
                "destination_tool": "tool_name",
                "reason": "reason for calling the tool",
                "parameters": {{
                    
                }}
            }}
        ]
    }}

    a. If all the required parameters are not available, or you want to clarify anything with the human customer, or you want the customer to 
    confirm the transaction, handoff to the **response_agent** 
    b. If all the required parameters are available, and the user has confirmed the transaction, call the appropriate tool(s)
    c. Only call the tool(s) if the user has confirmed the transaction. Users must explicitly confirm the transaction. Otherwise, you can continue the conversation with the customer by handoff to the **response_agent**
    d. When the tool call is successful, analyze the response and respond to the customer appropriately, or handoff to another agent if needed. 
    The response can be a failure or success. Remember, to respond to the user, simply handoff to yourself, the transaction agent, in the format below:
    e. If tool call is not sucessful apologize and ask the customer to try again.
    d. If the user does not confirm the transaction, either continue the conversation (e.g do you want another transaction?) or handoff to another agent if not able. 
 
    """  
    messages = [
        {"role": "system", "content": "You are a transaction agent that helps answer questions related to transactions, such as balance inquiries, transaction history, and account details, as well as perform transactions. You are part of a team of agents that make up Uliza, a helpful assistant that helps answer questions from Swiftcash bank customers."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        llm_response = json.loads(llm_response)
        
       
        
        
        # Update conversation conversation_history
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "Transaction_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": llm_response
        })

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response: {e}")
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content":  {{
                            "response_type": "handoff",
                            "agents": [
                                {
                                    "destination_agent": "response_agent",
                                    "reason": "Error parsing LLM response",
                                    "parameters": {{
                                        "response": "I apologize, but I encountered an error. Could you please rephrase your question?"
                                    }}
                                }
                            ]
                            }}})
        
        
    except Exception as e:
        state["conversation_history"].append({
            "role": "AI_AGENT",
            "node": "triage_agent",
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content":  {{
                            "response_type": "handoff",
                            "agents": [
                                {
                                    "destination_agent": "response_agent",
                                    "reason": "Error parsing LLM response",
                                    "parameters": {{
                                        "response": "I apologize, but I encountered an error. Please try again later."
                                    }}
                                }
                            ]
                            }}})

    return state

def send_money_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles fund transfer transactions.
    """
    transaction_status = 200  # Simulated success
    transaction_message = "Transaction successful"
    
    state["conversation_history"].append({
        "role": "AI_TOOL",
        "node": "send_money_tool",
        "conversation_id": state.get("conversation_id", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": {
            "transaction_status": transaction_status,
            "transaction_message": transaction_message,
            "request_parameters": state["conversation_history"][-1]["content"]
        }
    })

    return state

#empty state

#state = {}


if __name__ == "__main__":
    # Initialize variables
    user_input = "yes please"
    user_id = "123456"
    session_id = "7890"
    conversation_id = "112233"

    # Initialize state if it doesn't exist, otherwise keep existing state
    if 'state' not in locals():
        state = MainState(
            user_input="",
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            conversation_history=[],
            selected_tools=[],
            documents=[],
            tavily_results=[],
            final_response="",
            comprehensive_query="",
            similar_questions=[]
        )

    # Get existing conversation history
    existing_history = state.get("conversation_history", [])

    # Create the current message
    current_message = {
        "role": "human",
        "node": "human",
        "conversation_id": conversation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": user_input
    }

    # Update conversation history by appending the current message
    conversation_history = existing_history + [current_message]

    # Update state with new message while preserving existing history
    state.update({
        "user_input": user_input,
        "conversation_history": conversation_history
    })

    print("Current state:")
    print(json.dumps(state, indent=4))

    print("\nProcessing user input...")
    state = triage_agent(state)
    print("\nAfter handle_user_input:")
    print(json.dumps(state, indent=4))

    print("\nProcessing transaction...")
    state = transaction_agent(state)
    print("\nAfter transaction_agent:")
    print(json.dumps(state, indent=4))

    # Example of sending money
    if any(tool["name"] == "Account_transfer_tool" for tool in state.get("selected_tools", [])):
        print("\nExecuting money transfer...")
        state = send_money_tool(state)
        print(json.dumps(state, indent=4))
