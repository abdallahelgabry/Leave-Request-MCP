import os
import json
import asyncio
import streamlit as st
from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith import Client as LangSmithClient

langsmith_client = LangSmithClient()

load_dotenv()

# config
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OCI_CONFIG_PATH = os.getenv("OCI_CONFIG_PATH")
OCI_CONFIG_PROFILE = os.getenv("OCI_CONFIG_PROFILE")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
OCI_SERVICE_ENDPOINT = os.getenv("OCI_SERVICE_ENDPOINT")
OCI_COHERE_MODEL_ID = os.getenv("OCI_COHERE_MODEL_ID")

# LangSmith config
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# Wrap OpenAI client with LangSmith tracing
openai_client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY, timeout=60.0))



@traceable(name="load_prompt")
def load_system_prompt(prompt_name: str) -> str:
    """
    Load system prompt from LangSmith Prompt Hub
    """
    prompt = langsmith_client.pull_prompt(f"abdullah-elgabry/{prompt_name}")

    # Handle ChatPromptTemplate
    if hasattr(prompt, 'messages'):
        for msg in prompt.messages:
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, tuple) and msg[0] == 'system':
                return msg[1]
    
    return str(prompt)

# cohere client
def get_oci_client():
    """oracle cloud generative ai client"""
    try:
        import oci
        config = oci.config.from_file(OCI_CONFIG_PATH, OCI_CONFIG_PROFILE)
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=OCI_SERVICE_ENDPOINT,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize OCI client: {e}")
        return None


def convert_tools_to_cohere(tools: list) -> list:
    """convert tools format to cohere format"""
    import oci
    cohere_tools = []
    for tool in tools:
        func = tool["function"]
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        # parameter definitions
        param_defs = {}
        for param_name, param_info in properties.items():
            param_defs[param_name] = oci.generative_ai_inference.models.CohereParameterDefinition(
                type=param_info.get("type", "string"),
                description=param_info.get("description", ""),
                is_required=param_name in required
            )
        
        cohere_tools.append(
            oci.generative_ai_inference.models.CohereTool(
                name=func["name"],
                description=func["description"],
                parameter_definitions=param_defs if param_defs else None
            )
        )
    
    return cohere_tools


@traceable(name="cohere_chat_with_tools")
def call_cohere_with_tools(messages: list, tools: list, tool_results: list = None) -> dict:
    """call cohere model with tool"""
    import oci
    
    client = get_oci_client()
    if not client:
        return {"content": "Oracle Cloud connection failed.", "tool_calls": None}
    
    # extract system prompt and put history
    system_message = ""
    chat_history = []
    current_message = ""
    
    # extract system messages
    for msg in messages:
        if msg["role"] == "system":
            system_message += msg["content"] + "\n"
    
    # build chat history in order
    non_system_messages = [m for m in messages if m["role"] != "system"]
    
    # find the last user message for current message
    current_message = ""
    for msg in reversed(non_system_messages):
        if msg["role"] == "user" and msg.get("content"):
            current_message = msg["content"]
            break
    
    # all msg except the last one go to history
    for i, msg in enumerate(non_system_messages[:-1]):
        if msg["role"] == "user":
            chat_history.append(
                oci.generative_ai_inference.models.CohereUserMessage(
                    message=msg["content"]
                )
            )
        elif msg["role"] == "assistant" and msg.get("content"):
            chat_history.append(
                oci.generative_ai_inference.models.CohereChatBotMessage(
                    message=msg["content"]
                )
            )
    
    # convert tools to Cohere format
    cohere_tools = convert_tools_to_cohere(tools) if tools else None
    
    # convert tool results to cohere format if provided
    cohere_tool_results = None
    if tool_results:
        cohere_tool_results = []
        for tr in tool_results:
            cohere_tool_results.append(
                oci.generative_ai_inference.models.CohereToolResult(
                    call=oci.generative_ai_inference.models.CohereToolCall(
                        name=tr["name"],
                        parameters=tr.get("parameters", {})
                    ),
                    outputs=[{"result": tr["result"]}]
                )
            )
    
    # ensure message is never empty
    final_message = current_message.strip() if current_message else "Please continue."
    if not final_message:
        final_message = "Please continue."
    
    chat_request = oci.generative_ai_inference.models.CohereChatRequest(
        message=final_message,
        chat_history=chat_history if chat_history else None,
        preamble_override=system_message if system_message else None,
        tools=cohere_tools,
        tool_results=cohere_tool_results,
        is_force_single_step=True,
        max_tokens=1000,
        temperature=0.3
    )
    
    # build chat details
    chat_detail = oci.generative_ai_inference.models.ChatDetails(
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=OCI_COHERE_MODEL_ID
        ),
        compartment_id=OCI_COMPARTMENT_ID,
        chat_request=chat_request
    )
    
    try:
        response = client.chat(chat_detail)
        chat_response = response.data.chat_response
        
        # check for tool calls
        if hasattr(chat_response, 'tool_calls') and chat_response.tool_calls:
            tool_calls = []
            for i, tc in enumerate(chat_response.tool_calls):
                tool_calls.append({
                    "id": f"call_{i}",
                    "name": tc.name,
                    "parameters": tc.parameters if tc.parameters else {},
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.parameters) if tc.parameters else "{}"
                    }
                })
            return {
                "content": chat_response.text if hasattr(chat_response, 'text') else None,
                "tool_calls": tool_calls
            }
        
        return {
            "content": chat_response.text,
            "tool_calls": None
        }
        
    except Exception as e:
        st.error(f"Cohere API error: {e}")
        return {"content": f"Error: {str(e)}", "tool_calls": None}


# get mcp tools
@traceable(name="get_mcp_tools")
async def get_tools_from_server():
    """fetch available tools from mcp server"""
    try:
        async with Client(MCP_SERVER_URL) as client:
            tools = await client.list_tools()
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            return openai_tools
    except Exception as e:
        st.error(f"Cannot connect to MCP server: {e}")
        return []


@traceable(name="execute_mcp_tool")
async def execute_tool(tool_name: str, arguments: dict):
    """execute a tool on the mcp server"""
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(tool_name, arguments)
        if result.content:
            return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
        return str(result)


# openai chat
@traceable(name="chat_openai", run_type="chain")
async def chat_openai(message: str, conversation_history: list, employee_id: str):
    """process chat using openai"""
    
    tools = await get_tools_from_server()
    if not tools:
        return "Sorry, the leave request system is currently unavailable.", conversation_history
    
    system_prompt = load_system_prompt("leave-request-openai")  # hyphen not underscore


    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "system", "content": f"Current user's employee ID: {employee_id}"})
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    
    while assistant_message.tool_calls:
        messages.append(assistant_message)
        
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = await execute_tool(tool_name, arguments)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result if isinstance(result, str) else json.dumps(result)
            })
        
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message
    
    conversation_history.append({"role": "user", "content": message})
    conversation_history.append({"role": "assistant", "content": assistant_message.content})
    
    return assistant_message.content, conversation_history


# cohere chat
@traceable(name="chat_cohere", run_type="chain")
async def chat_cohere(message: str, conversation_history: list, employee_id: str):
    """process chat using cohere"""
    
    tools = await get_tools_from_server()
    if not tools:
        return "Sorry, the leave request system is currently unavailable.", conversation_history
    
    system_prompt = load_system_prompt("leave-request-cohere")
    messages = [{"role": "system", "content": system_prompt}]

    messages.append({"role": "system", "content": f"Employee ID: {employee_id}"})
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    # Call Cohere (first call without tool results)
    response = call_cohere_with_tools(messages, tools, tool_results=None)
    
    # Process tool calls
    while response.get("tool_calls"):
        # Execute all tools and collect results
        tool_results = []
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["name"]
            parameters = tool_call["parameters"]
            result = await execute_tool(tool_name, parameters)
            
            tool_results.append({
                "name": tool_name,
                "parameters": parameters,
                "result": result if isinstance(result, str) else json.dumps(result)
            })
        
        # Call Cohere again with tool results
        response = call_cohere_with_tools(messages, tools, tool_results=tool_results)
    
    final_content = response.get("content", "Sorry, I couldn't process your request.")
    
    conversation_history.append({"role": "user", "content": message})
    conversation_history.append({"role": "assistant", "content": final_content})
    
    return final_content, conversation_history


@traceable(name="chat_router", run_type="chain")
async def chat(message: str, conversation_history: list, employee_id: str, model_provider: str):
    """route chat based on model"""
    if model_provider == "OpenAI":
        return await chat_openai(message, conversation_history, employee_id)
    else:
        return await chat_cohere(message, conversation_history, employee_id)


def run_async(coro):
    """run async functions"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


#streamlit

st.set_page_config(page_title="Leave Request Assistant", page_icon="📅")

st.title("📅 Leave Request Assistant")

# Initialize session state first
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "current_model" not in st.session_state:
    st.session_state.current_model = None

# Sidebar
with st.sidebar:
    
    # model Selection
    st.subheader("Model Provider")
    model_provider = st.radio(
        "Choose AI Model:",
        ["OpenAI", "Cohere"],
        index=0
    )
    
    # check if model changed clear chat if so
    if st.session_state.current_model is not None and st.session_state.current_model != model_provider:
        st.session_state.messages = []
        st.session_state.history = []
    
    # update current model
    st.session_state.current_model = model_provider

    
    st.markdown("---")
    
    # emp id
    st.subheader("Employee")
    employee_id = st.text_input("Employee ID", value="EMP001")
    
    
    st.markdown("---")
    st.markdown("**Test Employee IDs:**")
    st.markdown("- EMP001")
    st.markdown("- EMP002")
    st.markdown("- EMP003")

# display chat msg
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    # add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant placeholder
    assistant_placeholder = st.chat_message("assistant")
    with assistant_placeholder:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking...")

    response, st.session_state.history = run_async(
        chat(prompt, st.session_state.history, employee_id, model_provider)
    )
    thinking_placeholder.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )