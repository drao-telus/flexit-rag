import json


def calculate_tip(bill_amount, tip_percentage=10):
    """
    Calculate tip and total amount

    Args:
        bill_amount (float): The bill amount in dollars
        tip_percentage (float): Tip percentage (default: 10)

    Returns:
        dict: Dictionary containing bill amount, tip percentage, tip amount, and total amount
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount

    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip_amount, 2),
        "total_amount": round(total_amount, 2),
    }


# Function schema for OpenAI function calling
CALCULATE_TIP_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_tip",
        "description": "Calculate tip and total amount for a bill. Default tip is 10% if not specified.",
        "parameters": {
            "type": "object",
            "properties": {
                "bill_amount": {
                    "type": "number",
                    "description": "The bill amount in dollars",
                },
                "tip_percentage": {
                    "type": "number",
                    "description": "Tip percentage (default: 10)",
                    "default": 10,
                },
            },
            "required": ["bill_amount"],
        },
    },
}

# Function dispatcher to handle function calls
AVAILABLE_FUNCTIONS = {"calculate_tip": calculate_tip}


def execute_function_call(function_name, arguments):
    """
    Execute a function call with the given arguments

    Args:
        function_name (str): Name of the function to call
        arguments (dict): Arguments to pass to the function

    Returns:
        dict: Function result or error message
    """
    try:
        if function_name in AVAILABLE_FUNCTIONS:
            function = AVAILABLE_FUNCTIONS[function_name]
            result = function(**arguments)
            return {"success": True, "result": result}
        else:
            return {"success": False, "error": f"Function '{function_name}' not found"}
    except Exception as e:
        return {
            "success": False,
            "error": f"Error executing function '{function_name}': {str(e)}",
        }


def format_function_result(function_name, result):
    """
    Format function result for display

    Args:
        function_name (str): Name of the function
        result (dict): Function execution result

    Returns:
        str: Formatted result string
    """
    if not result.get("success"):
        return f"❌ Error: {result.get('error', 'Unknown error')}"

    if function_name == "calculate_tip":
        data = result["result"]
        return f"""
💰 **Tip Calculation Result**
- Bill Amount: ${data["bill_amount"]:.2f}
- Tip Percentage: {data["tip_percentage"]}%
- Tip Amount: ${data["tip_amount"]:.2f}
- **Total Amount: ${data["total_amount"]:.2f}**
"""

    return f"✅ Function '{function_name}' executed successfully: {json.dumps(result['result'], indent=2)}"


# List of all available tools for assistant modification
ALL_FUNCTION_TOOLS = [CALCULATE_TIP_TOOL]
