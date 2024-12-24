from Shared_services.logging import setup_logger

logger = setup_logger()

def complaints_tool(user_input: str, conversation_history: list, kyc: list) -> dict:
    """Dummy tool for complaints"""
    logger.info(f"Complaint logged successfully: {user_input}")
    return {"status": "success", "message": "Complaint logged successfully"}

def human_handoff_tool(user_input: str, conversation_history: list, kyc: list) -> dict:
    """Dummy tool for human handoff"""
    logger.info(f"Human handoff successful: {user_input}")
    return {"status": "success", "message": "Human handoff successful"}
