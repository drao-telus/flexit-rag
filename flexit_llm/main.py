import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv
from functions import (
    CALCULATE_TIP_TOOL,
    calculate_tip,
    format_function_result,
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Function Calling Demo",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def make_function_call_request(messages, tools, model="gpt-4o"):
    """
    Make a request to the Fuelix API with function calling

    Args:
        messages: List of message dictionaries
        tools: List of function tools
        model: Model name to use

    Returns:
        dict: API response data or None if error
    """
    try:
        url = "https://api.fuelix.ai/v1/chat/completions"

        payload = {
            "messages": messages,
            "model": model,
            "tools": tools,
            "tool_choice": "auto",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('FUELIX_API_KEY')}",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(
                f"API request failed with status {response.status_code}: {response.text}"
            )
            return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def execute_tool_calls(tool_calls):
    """
    Execute tool calls and return results

    Args:
        tool_calls: List of tool calls from the API response

    Returns:
        list: List of tool call results
    """
    results = []

    for tool_call in tool_calls:
        function_name = tool_call.get("function", {}).get("name")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        tool_call_id = tool_call.get("id")

        try:
            arguments = json.loads(arguments_str)

            if function_name == "calculate_tip":
                result = calculate_tip(**arguments)
                results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result),
                    }
                )

                # Display the function call and result
                st.write(f"🔧 **Function Call:** `{function_name}`")
                st.write(f"📝 **Arguments:** `{arguments}`")

                formatted_result = format_function_result(
                    function_name, {"success": True, "result": result}
                )
                st.markdown(formatted_result)

            else:
                st.error(f"Unknown function: {function_name}")

        except json.JSONDecodeError:
            st.error(f"Invalid JSON arguments for function {function_name}")
        except Exception as e:
            st.error(f"Error executing function {function_name}: {str(e)}")

    return results


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = bool(os.getenv("FUELIX_API_KEY"))


def display_chat_messages():
    """Display all chat messages"""
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def main():
    """Main Streamlit application"""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🧮 Function Calling Demo")
        st.markdown("---")

        # API Key status
        if st.session_state.api_key_valid:
            st.success("✅ API Key loaded")
        else:
            st.error("❌ API Key not found")
            st.info("Please set FUELIX_API_KEY in your .env file")

        # Model selection
        model_options = ["gpt-4o", "gpt-4", "claude-3-5-sonnet"]
        selected_model = st.selectbox("Select Model", model_options, index=0)

        st.markdown("---")

        # Function info
        st.header("🔧 Available Functions")
        st.write("**calculate_tip**")
        st.write("- Calculate tip and total amount")
        st.write("- Default tip: 10%")
        st.write("- Parameters: bill_amount, tip_percentage (optional)")

        st.markdown("---")

        # Quick test
        st.header("🧪 Quick Test")

        test_col1, test_col2 = st.columns(2)
        with test_col1:
            bill_amount = st.number_input(
                "Bill ($)", min_value=0.01, value=50.0, step=0.01
            )
        with test_col2:
            tip_percentage = st.number_input(
                "Tip (%)", min_value=0.0, value=10.0, step=0.5
            )

        if st.button("🧮 Test Function", use_container_width=True):
            result = calculate_tip(bill_amount, tip_percentage)
            formatted_result = format_function_result(
                "calculate_tip", {"success": True, "result": result}
            )
            st.markdown(formatted_result)

        st.markdown("---")

        # Chat controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Example prompts
        st.header("💡 Try These Examples")
        example_prompts = [
            "Calculate tip for $75 with 18%",
            "What's the tip for a $25 bill?",
            "I need to tip 20% on $120",
            "Calculate tip for $89.50 with 15%",
        ]

        for prompt in example_prompts:
            if st.button(f"💬 {prompt}", use_container_width=True):
                # Add the example prompt to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    # Main chat interface
    st.title("💬 Chat with Function Calling")
    st.markdown("Ask me to calculate tips and I'll use the `calculate_tip` function!")

    # Display existing messages
    display_chat_messages()

    # Chat input
    if prompt := st.chat_input(
        "Ask me to calculate a tip... (e.g., 'Calculate tip for $50')"
    ):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            if not st.session_state.api_key_valid:
                st.error("Please configure your API key in the .env file")
                return

            # Show thinking indicator
            with st.spinner("Processing..."):
                # Prepare messages for API call
                api_messages = [
                    msg
                    for msg in st.session_state.messages
                    if msg["role"] in ["user", "assistant", "tool"]
                ]

                # Add system message for function calling
                system_message = {
                    "role": "system",
                    "content": "You are a helpful assistant that can calculate tips. When users ask about tip calculations, use the calculate_tip function. Always be friendly and explain the calculation clearly.",
                }
                api_messages.insert(0, system_message)

                # Define available tools
                tools = [CALCULATE_TIP_TOOL]

                # Make API request
                response = make_function_call_request(
                    api_messages, tools, selected_model
                )

                if response and "choices" in response:
                    choice = response["choices"][0]
                    message = choice["message"]

                    # Check if there are tool calls
                    tool_calls = message.get("tool_calls")

                    if tool_calls:
                        # Add the assistant's message with tool calls to history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": message.get("content") or "",
                                "tool_calls": tool_calls,
                            }
                        )

                        # Execute tool calls
                        tool_results = execute_tool_calls(tool_calls)

                        # Add tool results to messages
                        st.session_state.messages.extend(tool_results)

                        # Make another API call to get the final response
                        final_messages = [
                            msg
                            for msg in st.session_state.messages
                            if msg["role"] in ["user", "assistant", "tool"]
                        ]
                        final_messages.insert(0, system_message)

                        final_response = make_function_call_request(
                            final_messages, tools, selected_model
                        )

                        if final_response and "choices" in final_response:
                            final_message = final_response["choices"][0]["message"]
                            assistant_reply = final_message.get("content", "")

                            if assistant_reply:
                                st.markdown(assistant_reply)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": assistant_reply}
                                )
                        else:
                            st.error(
                                "Failed to get final response after function execution"
                            )

                    else:
                        # No tool calls, just display the regular response
                        assistant_reply = message.get("content", "")
                        if assistant_reply:
                            st.markdown(assistant_reply)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": assistant_reply}
                            )
                        else:
                            st.error("No response content received")

                    # Show token usage if available
                    if "usage" in response:
                        usage = response["usage"]
                        st.caption(
                            f"Tokens used: {usage.get('total_tokens', 'N/A')} "
                            f"(Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                            f"Completion: {usage.get('completion_tokens', 'N/A')})"
                        )
                else:
                    error_message = (
                        "Sorry, I couldn't generate a response. Please try again."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )


if __name__ == "__main__":
    main()
