from openai import OpenAI
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def call_llm_api(messages: List[Dict[str, str]], 
                model: str = "gpt-4o-mini-2024-07-18",
                max_tokens: int = 2000,
                temperature: float = 0.3) -> Any:
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

# Alternative implementation using OpenRouter
def call_llm_api_openrouter(messages: List[Dict[str, str]], 
                model: str = "google/gemini-2.0-flash-thinking-exp:free",
                max_tokens: int = 2000,
                temperature: float = 0.3) -> Optional[str]:
    """
    Make a call to the OpenRouter API for chat completions.
    Returns the response content or None if there's an error.s
    """
    try:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        #print(f"Sending request to model: {model}")  # Debug line
        #print(f"Messages: {messages}")  # Debug line
        
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if not response or not response.choices:
            print("No response or choices from API")
            return None
            
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in OpenRouter API call: {e}")
        print(f"Full error details: {traceback.format_exc()}")  # Add full traceback
        return None  # Return None instead of raising
