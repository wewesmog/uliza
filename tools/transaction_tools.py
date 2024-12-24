
import random
from datetime import datetime, timezone, timedelta
from shared_services.logger_setup import setup_logger

logger = setup_logger()
# Constants for status codes and messages
SUCCESS_CODE = "SUCCESS_001"
ERROR_CODE = "ERROR_001"

# Customer-friendly messages
MESSAGES = {
    "account_transfer_tool": {
        "success": "Your transfer of KES {amount} to account {account_number_to} was successful. Your new balance is KES {balance}.",
        "error": "We couldn't complete your transfer at this time. Please try again later or contact support."
    },
    "balance_inquiry_tool": {
        "success": "Your available balance is KES {available_balance}.",
        "error": "We couldn't fetch your balance at this time. Please try again later."
    },
    "transaction_history_tool": {
        "success": "Here are your last 5 transactions. Your current balance is KES {current_balance}.",
        "error": "We couldn't retrieve your transaction history at this time. Please try again later."
    },
    "paybill_tool": {
        "success": "Your payment of KES {amount} to {biller_code} was successful. Reference number: {receipt_number}.",
        "error": "We couldn't complete your bill payment at this time. Please try again later."
    }
}

def account_transfer_tool(account_number_from: str, account_number_to: str, amount: str, 
                         reason: str = None, confirm: bool = True) -> dict:
    """Dummy tool for account transfers"""
    try:
        balance = str(random.randint(1000, 100000))
        response = {
            "tool_name": "account_transfer_tool",
            "status": "success",
            "code": SUCCESS_CODE,
            "message": MESSAGES["account_transfer_tool"]["success"].format(
                amount=amount,
                account_number_to=account_number_to,
                balance=balance
            ),
            "transaction_id": "TRX" + str(random.randint(100000, 999999)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "from_account": account_number_from,
                "to_account": account_number_to,
                "amount": amount,
                "reason": reason,
                "balance": balance
            }
        }
    except Exception as e:
        response = {
            "status": "error",
            "code": ERROR_CODE,
            "message": MESSAGES["account_transfer_tool"]["error"],
            "error_details": str(e)
        }
    
    return response

def balance_inquiry_tool(account_number: str, reason: str = None, confirm: bool = True) -> dict:
    """Dummy tool for balance inquiries"""
    try:
        available_balance = str(random.randint(1000, 100000))
        response = {
            "tool_name": "balance_inquiry_tool",
            "status": "success",
            "code": SUCCESS_CODE,
            "message": MESSAGES["balance_inquiry_tool"]["success"].format(
                available_balance=available_balance
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "account_number": account_number,
                "available_balance": available_balance,
                "actual_balance": str(random.randint(1000, 100000))
            }
        }
    except Exception as e:
        response = {
            "status": "error",
            "code": ERROR_CODE,
            "message": MESSAGES["balance_inquiry_tool"]["error"],
            "error_details": str(e)
        }
    return response

def transaction_history_tool(account_number: str, reason: str = None, confirm: bool = True) -> dict:
    """Dummy tool for transaction history"""
    try:
        current_balance = str(random.randint(1000, 100000))
        transactions = [
            {
                "date": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                "type": random.choice(["CREDIT", "DEBIT"]),
                "amount": str(random.randint(100, 10000)),
                "description": random.choice(["SALARY", "TRANSFER", "ATM", "PAYMENT"]),
                "balance": str(random.randint(1000, 100000))
            } for i in range(5)
        ]
        
        response = {
            "tool_name": "transaction_history_tool",
            "status": "success", 
            "code": SUCCESS_CODE,
            "message": MESSAGES["transaction_history_tool"]["success"].format(
                current_balance=current_balance
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "account_number": account_number,
                "current_balance": current_balance,
                "transactions": transactions
            }
        }
    except Exception as e:
        response = {
            "status": "error",
            "code": ERROR_CODE,
            "message": MESSAGES["transaction_history_tool"]["error"],
            "error_details": str(e)
        }
    return response

def paybill_tool(biller_code: str, account_number: str, amount: str, 
                 reason: str = None, confirm: bool = True) -> dict:
    """Dummy tool for bill payments"""
    try:
        receipt_number = "RCP" + str(random.randint(100000, 999999))
        response = {
            "tool_name": "paybill_tool",
            "status": "success",
            "code": SUCCESS_CODE,
            "message": MESSAGES["paybill_tool"]["success"].format(
                amount=amount,
                biller_code=biller_code,
                receipt_number=receipt_number
            ),
            "transaction_id": "BIL" + str(random.randint(100000, 999999)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "biller_code": biller_code,
                "account_number": account_number,
                "amount": amount,
                "reason": reason,
                "receipt_number": receipt_number
            }
        }
    except Exception as e:
        response = {
            "status": "error",
            "code": ERROR_CODE,
            "message": MESSAGES["paybill_tool"]["error"],
            "error_details": str(e)
        }
    
    return response


# Dictionary mapping tool names to their functions
TOOLS = {
    "account_transfer_tool": account_transfer_tool,
    "balance_inquiry_tool": balance_inquiry_tool,
    "transaction_history_tool": transaction_history_tool,
    "paybill_tool": paybill_tool
}