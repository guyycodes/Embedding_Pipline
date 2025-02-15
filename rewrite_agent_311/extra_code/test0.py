import os
import json
import requests
from typing import Dict, Any
from pydantic import Field

# Example LLM & messages stubs (you can adapt to real classes)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# LangGraph for building the agent state machine
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

##############################################################################
# 1. Define Tools
##############################################################################

def get_regulations(endpoint: str) -> Dict[str, Any]:
    """
    Attempt to GET regulations from the provided endpoint.
    Returns a JSON dict:
      {
        "success": bool,
        "data": <raw_data_if_success> or None,
        "error": <error_message_if_failure> or None
      }
    """
    try:
        resp = requests.get(endpoint, timeout=5)
        if resp.status_code == 200:
            return {
                "success": True,
                "data": resp.text,   # or resp.json() if you want structured data
                "error": None
            }
        else:
            return {
                "success": False,
                "data": None,
                "error": f"HTTP {resp.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

def create_pdf(raw_data: str) -> Dict[str, Any]:
    """
    Convert raw data (string or JSON) to a PDF file locally.
    Return a dict: { "file_path": "...some_path..." }.
    Here we mock it by writing to a temporary file.
    """
    pdf_path = "regulations.pdf"
    # Mock PDF creation: just write text to a file with .pdf extension
    # In real usage, you'd generate an actual PDF (with a library like ReportLab).
    with open(pdf_path, "w", encoding="utf-8") as f:
        f.write("Fake PDF Content:\n")
        f.write(raw_data)
    return {"file_path": pdf_path}

def set_regulations(file_path: str) -> Dict[str, Any]:
    """
    Upload the PDF to a known ingestion endpoint.
    For demonstration, we do a simple multipart/form-data POST.
    """
    url = "http://127.0.0.1:8675/api/upload/documents"
    try:
        with open(file_path, "rb") as pdf:
            resp = requests.post(
                url,
                files={"file": pdf}  # or "file": ("filename.pdf", pdf, "application/pdf")
            )
        if resp.status_code == 200:
            return {
                "success": True,
                "message": f"File {file_path} uploaded successfully."
            }
        else:
            return {
                "success": False,
                "message": f"Upload failed with code {resp.status_code}"
            }
    except Exception as e:
        return {"success": False, "message": str(e)}

# Gather the tool references
tools_list = [get_regulations, create_pdf, set_regulations]

##############################################################################
# 2. Set up LLM with Tools
##############################################################################

# System instructions: you can add more guardrails or guidance here
system_message = SystemMessage(content="""You are a helpful AI assistant.

You have the following tools at your disposal:
1. get_regulations(endpoint: str)
2. create_pdf(raw_data: str)
3. set_regulations(file_path: str)

Behavior:
- If the user asks to retrieve or update regulations from a specific endpoint, 
  you may call 'get_regulations' with that URL.
- If 'get_regulations' fails, you may retry up to 3 total times.
- Once you have data, if appropriate, call 'create_pdf' to create a PDF from the data.
- Finally, you may call 'set_regulations' to upload the PDF.
- If the user does not request any data retrieval, you may simply respond without using any tools.
- End after you have either retried too many times or successfully uploaded.
""")

# A minimal custom state for demonstration
class CustomState(MessagesState):
    attempt_count: int = Field(0, description="Count how many times we've called get_regulations")
    max_attempts: int = Field(3, description="Max attempts for get_regulations")

llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools_list)


##############################################################################
# 3. Define Node Functions
##############################################################################

def assistant_node(state: CustomState):
    """
    The main 'assistant' node in the graph. Each pass:
      1) We feed the system + conversation so far to the LLM.
      2) The LLM either returns a normal message or calls one of the 3 tools.
    """
    # If we've gone beyond normal iteration (just a safeguard),
    # you could stop here. Or remove this check if you don't want a limit.
    if len(state["messages"]) > 20:
        return {"messages": [{"role": "assistant", "content": "Max turns reached."}]}
    
    # Let the LLM produce a response (or a function call)
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def get_regulations_tool_node(state: CustomState):
    """
    Actually call get_regulations with the 'endpoint' arg from the LLM message.
    Check success/failure. If failure and attempt_count < max_attempts, 
    the LLM might decide to call it again.
    """
    # The last tool call arguments
    last_msg = state["messages"][-1]
    args = last_msg.get("arguments", {})
    endpoint = args.get("endpoint", "")

    # Count attempt
    state.attempt_count += 1

    # Execute tool
    result = get_regulations(endpoint)

    # Return as tool message
    return {
        "messages": [
            {"role": "tool", "content": json.dumps(result)}
        ]
    }


def create_pdf_tool_node(state: MessagesState):
    """
    Call create_pdf with the raw_data from the LLM's function-call arguments.
    """
    last_msg = state["messages"][-1]
    args = last_msg.get("arguments", {})
    raw_data = args.get("raw_data", "")

    result = create_pdf(raw_data)
    return {"messages": [{"role": "tool", "content": json.dumps(result)}]}


def set_regulations_tool_node(state: MessagesState):
    """
    Call set_regulations with a file_path from the LLM's function-call arguments.
    """
    last_msg = state["messages"][-1]
    args = last_msg.get("arguments", {})
    file_path = args.get("file_path", "")

    result = set_regulations(file_path)
    return {"messages": [{"role": "tool", "content": json.dumps(result)}]}


##############################################################################
# 4. Decide Which Node to Call Next (Conditional Logic)
##############################################################################

def call_tools_condition(state: CustomState, node_result):
    """
    Inspects the LLM's last output (node_result["messages"][-1]) to see
    if it wants to call a tool. If so, route to the correct node.
    Otherwise, route to 'END'.
    
    Also handle the 'get_regulations' attempt limit:
      - If tool is 'get_regulations' and we've reached 3 attempts, go to END.
    """
    last_msg = node_result["messages"][-1]

    tool_name = last_msg.get("name", None)
    if not tool_name:
        # No function call => no tool => end
        return "END"
    
    if tool_name == "get_regulations":
        # Check attempt limit
        if state.attempt_count >= state.max_attempts:
            # We've already tried too many times => do not route to the tool again
            return "END"
        return "get_regulations_node"
    
    elif tool_name == "create_pdf":
        return "create_pdf_node"
    
    elif tool_name == "set_regulations":
        return "set_regulations_node"
    
    # If some unknown tool is requested, just end
    return "END"


##############################################################################
# 5. Build and Compile the Graph
##############################################################################

builder = StateGraph(CustomState)

# Node: "assistant_node" => produces either a normal message or a function-call
builder.add_node("assistant_node", assistant_node)

# Node: "get_regulations_node" => calls get_regulations tool
builder.add_node("get_regulations_node", get_regulations_tool_node)

# Node: "create_pdf_node" => calls create_pdf tool
builder.add_node("create_pdf_node", create_pdf_tool_node)

# Node: "set_regulations_node" => calls set_regulations tool
builder.add_node("set_regulations_node", set_regulations_tool_node)

# Start => assistant
builder.add_edge(START, "assistant_node")

# After assistant => decide if we want a tool
builder.add_conditional_edges("assistant_node", call_tools_condition)

# The tool nodes, once done, go back to the assistant for further instructions
builder.add_edge("get_regulations_node", "assistant_node")
builder.add_edge("create_pdf_node", "assistant_node")
builder.add_edge("set_regulations_node", "assistant_node")

# Compile the final state machine
graph = builder.compile()

##############################################################################
# 6. Usage Example
##############################################################################

if __name__ == "__main__":
    # Example: user wants to fetch data from an endpoint
    initial_state = CustomState(messages=[
        {"role": "user", "content": "Please retrieve new regulations from https://example.com/api/regulations"},
    ])

    final_state = graph.run(initial_state)
    
    print("\n=== Conversation Trace ===\n")
    for idx, msg in enumerate(final_state["messages"], 1):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "assistant":
            print(f"Assistant: {content}")
        elif role == "user":
            print(f"User: {content}")
        elif role == "tool":
            print(f"[Tool Output]: {content}")
        else:
            print(f"{role.title()}: {content}")
