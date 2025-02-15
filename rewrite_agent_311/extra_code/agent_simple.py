# My Code
import os
import getpass
from dotenv import load_dotenv
import requests
from typing import Tuple
from typing_extensions import TypedDict

##############################
# LangGraph & LangChain Imports
##############################
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode  # important

##############################
# 0. Environment & Setup 
##############################
load_dotenv()

def _set_env(var: str):
    """
    Prompt for environment variables if not found.
    Only relevant if you need interactive usage.
    """
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")

# Turn on optional tracing if needed:
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "")
os.environ["LANGSMITH_ENDPOINT"]

##############################
# 1. State Definition
##############################
class MessageDict(TypedDict):
    role: str
    content: str

class RewriteFlowState(TypedDict):
    """
    A custom state to keep track of all necessary data for our flow.
    We are storing messages as plain dictionaries (role/content).
    """
    messages: list[MessageDict]
    user_query: str
    requires_external_data: bool
    # If the rewrite agent calls the vector store, we set this to True
    vector_store_checked: bool = False
    # The raw or summarized vector store result (if any)
    vector_store_answer: str = ""
    
##############################
# Helpers to convert AIMessage -> dict
##############################
def msg_to_dict(msg) -> MessageDict:
    """
    Convert a LangChain message (AIMessage, HumanMessage, SystemMessage)
    into a typed dict with role, content.
    """
    if isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    elif isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, dict):
        # Already a dict, presumably with "role" / "content"
        return msg
    else:
        # Fallback
        return {"role": "assistant", "content": str(msg)}

##############################
# 2. LLM Definitions
##############################
rewrite_model = ChatOpenAI(
    model="gpt-4",  # Using GPT-4 for rewriting
    temperature=0
)

response_model = ChatOpenAI(
    model="gpt-4o",  # Using a hypothetical "gpt-4o" endpoint
    temperature=0
)

##############################
# 3. Tools
##############################
def vector_store_tool(optimized_query: str) -> Tuple[str, int]:
    """
    Low-level function that actually calls the vector store endpoint.
    Returns (answer_string, status_code).
    """
    url = "http://localhost:8675/api/models/qa"
    payload = {"query": optimized_query}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            json_data = response.json()
            answer = json_data.get("answer", "")
            return answer, 200
        else:
            return f"Vector store lookup failed. Code: {response.status_code}", response.status_code
    except Exception as e:
        return f"Exception calling QA endpoint: {str(e)}", 500

def vector_store_lookup(query: str) -> str:
    """
    This is the high-level *tool* that the LLM can call. 
    It returns just a string (either the raw answer or an error).
    """
    answer, code = vector_store_tool(query)
    if code == 200:
        return answer
    else:
        return f"[Vector store error: {code}] {answer}"

##############################
# 4. Bind Tools to the Rewrite LLM
##############################
rewrite_model_with_tools = rewrite_model.bind_tools([vector_store_lookup])

##############################
# 5. Node Implementations
##############################

def rewrite_agent_node(state: RewriteFlowState):
    """
    The multi-step "Rewrite Agent" node.

    On each pass:
      - We have a system prompt that instructs the LLM it may call vector_store_lookup(...) if needed.
      - If the LLM *already* has some vector_store_answer in state, we include it in the conversation so 
        the LLM can summarize it on this pass.
      - If the user query might require external scanning, the LLM can also decide to set requires_external_data=True
        (meaning we route to scanner_agent_node).
      - Finally, it returns a JSON structure:
        {
          "requires_external_data": bool,
          "optimized_query": "..."
        }
      which we parse to update the state.
    """
    # 1. Ensure user_query is in state (pull from latest user message if needed)
    if not state["user_query"]:
        last_user_msg = next(
            (msg for msg in reversed(state["messages"]) if msg["role"] == "user"), 
            None
        )
        if last_user_msg:
            state["user_query"] = last_user_msg["content"]
        else:
            state["user_query"] = "No user query provided."

    # 2. Build a system prompt that:
    #    - Mentions the available tool
    #    - Tells the LLM to produce final JSON with requires_external_data & optimized_query
    #    - (Optionally) instructs it to incorporate or summarize the vector_store_answer if state.vector_store_checked is True
    system_instructions = f"""You are the Rewrite Agent. You have one tool available:
1) vector_store_lookup(query: str) -> str

Your overall goals:
 - Possibly refine the user's query if needed for vector store usage.
 - If needed, call vector_store_lookup(...) to gather additional context from the vector store with the standard "tool call" format.
 - If you require *other* external data from an external API or website, set "requires_external_data" to true.
 - Summarize any relevant vector store data you have (if any).
 - Finally, output one *final* JSON message (in your assistant role) with the structure:
   {{
     "requires_external_data": boolean,
     "optimized_query": "some string"
   }}

IMPORTANT details:
 - If you decide to call the vector store tool, do so with the standard "tool call" format. 
 - You may call the vector store tool multiple times if needed.
 - Once you produce the final JSON, do not call any more tools. That JSON must be your last assistant message.
 - If 'vector_store_answer' is provided below, you have already fetched data from the vector store. Summarize it.

Below is your *current known data*:
 - user_query: {state["user_query"]}
 - vector_store_checked: {state["vector_store_checked"]}
 - vector_store_answer: {state["vector_store_answer"]}
"""

    # Combine with existing conversation
    conversation = [
        SystemMessage(content=system_instructions)
    ]
    # Convert existing dict messages to the appropriate LLM "Message" objects:
    for msg in state["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            conversation.append(SystemMessage(content=content))
        elif role == "assistant":
            conversation.append(AIMessage(content=content))
        else:  # default to 'user'
            conversation.append(HumanMessage(content=content))

    # 3. Invoke the LLM with tools
    final_response_msg = rewrite_model_with_tools.invoke(conversation)

    # Convert the returned AIMessage to a dict
    new_msg_dict = msg_to_dict(final_response_msg)

    # 4. Append to state
    state["messages"].append(new_msg_dict)

    # 5. Attempt to parse final JSON
    import json
    requires_external_data = False
    optimized_query = state["user_query"]
    try:
        parsed = json.loads(final_response_msg.content)
        requires_external_data = parsed.get("requires_external_data", False)
        optimized_query = parsed.get("optimized_query", optimized_query)
    except:
        pass
    node_result = {
        "messages": [new_msg_dict],
        "optimized_query": optimized_query
        # possibly other data
    }
    

    # 6. Update state
    state["requires_external_data"] = requires_external_data
    # Save it so the routing function can read it.
    state["last_node_result"] = node_result

    # 7. Return partial updates for the node
    return node_result


def rewrite_tools_node(state: RewriteFlowState):
    """
    This node executes the tool call whenever the LLM
    tries to call "vector_store_lookup(...)".

    We capture the tool's result, set vector_store_checked=True,
    and store the result in vector_store_answer.
    """
    # 1) Parse & execute the LLM's requested tool call
    toolnode_output = ToolNode([vector_store_lookup])(state)
    # toolnode_output might look like:
    # {
    #   "messages": [AIMessage(...), ...],
    #   "tool_call": {...}
    # }

    # 2) Convert all returned messages into dicts
    converted_messages = []
    for m in toolnode_output["messages"]:
        converted_messages.append(msg_to_dict(m))
    
    # 3) Append them all to our conversation in state
    state["messages"].extend(converted_messages)

    # 4) Mark that we have used the vector store
    #    and store the last tool output into state["vector_store_answer"]
    if converted_messages:
        last_msg = converted_messages[-1]
        state["vector_store_answer"] = last_msg["content"]
    state["vector_store_checked"] = True

    # 5) Return these messages to the graph
    return {
        "messages": converted_messages
    }

def scanner_agent_node(state: RewriteFlowState, node_result):
    """
    If requires_external_data is set, we come here. 
    In a future version, we might do external scanning or external API calls.
    For now, it just returns a placeholder message.
    """
    placeholder_result = "Scanner Agent Placeholder - possibly calling external APIs or undisclosed tools next."
    return {
        "messages": [
            {"role": "assistant", "content": placeholder_result}
        ]
    }
    
def undisclosed_tool_call_node(state: RewriteFlowState):
    """
    Placeholder node for an undisclosed tool call or final scanning step.
    After this node, we go to END (no direct user response).
    """
    # Potentially call some hidden function or do some final tasks here.
    return {
        "messages": [
            {"role": "assistant", "content": "Undisclosed tool call complete. No user-facing response."}
        ]
    }

def response_agent_node(state: RewriteFlowState):
    """
    Final node that uses GPT-4o to produce a final answer to the user.
    We pass in the original user_query plus the (summarized) vector_store_answer.
    """
    user_query = state["user_query"]
    vs_answer = state["vector_store_answer"] or ""

    sys_prompt = """You are the final Response Agent. 
You receive the user's original query and any available vector store data.
If there's NO vector store data, you may ask clarifying questions, unless you are confident to answer.
If there IS vector store data, incorporate it into your final answer.
Never mention using a vector store or an external tool.
"""

    content_msg = f"User query: {user_query}\nVector Store Data: {vs_answer}"

    # In-memory conversation:
    conversation = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=content_msg)
    ]
    final_answer = response_model.invoke(conversation)

    # Convert AIMessage to dict
    new_msg_dict = {
        "role": final_answer.role,    # typically 'assistant'
        "content": final_answer.content
    }

    return {
        "messages": [new_msg_dict]
    }

##############################
# 6. Routing Logic
##############################
def route_from_rewrite(state: RewriteFlowState):
    """
    Route logic:
      - If requires_external_data => scanner_agent_node
      - Else if vector_store_checked == False => rewrite_tools_node
      - Else => response_agent_node
    """
    if state["requires_external_data"]:
        return "scanner_agent_node"
    elif not state["vector_store_checked"]:
        return "rewrite_tools_node"
    else:
        return "response_agent_node"

##############################
# 7. CUSTOM ROUTING FOR TOOLS vs. "Fallback"
##############################
def rewrite_routing_function(state: RewriteFlowState):
    """
    We check if the LLM's latest message was a tool call using tools_condition.
    If so, route to the rewrite_tools_node.
    Otherwise, fallback to route_from_rewrite.
    """
    node_result = state.get("last_node_result", {})  # fallback to {}
    next_node = tools_condition(state, node_result)
    if next_node is not None:
        return "rewrite_tools_node"
    else:
        return route_from_rewrite(state)

##############################
# 8. Building the Graph
##############################
builder = StateGraph(RewriteFlowState)

# Main rewrite agent node
builder.add_node("rewrite_agent_node", rewrite_agent_node)

# The dedicated tool execution node
builder.add_node("rewrite_tools_node", rewrite_tools_node)

# Scanner, undisclosed tool, and response
builder.add_node("scanner_agent_node", scanner_agent_node)
builder.add_node("undisclosed_tool_call_node", undisclosed_tool_call_node)
builder.add_node("response_agent_node", response_agent_node)

# 1) Start -> rewrite
builder.add_edge(START, "rewrite_agent_node")

# 2) If the rewrite agent decides to call a tool => "rewrite_tools_node", else => route_from_rewrite
builder.add_conditional_edges("rewrite_agent_node", rewrite_routing_function)

# 3) After tool executes, come back to rewrite
builder.add_edge("rewrite_tools_node", "rewrite_agent_node")

# 4) From rewrite_agent_node => route_from_rewrite => either "scanner_agent_node" or "response_agent_node"
#    (No direct edge to END from rewrite_agent_node.)

# 5) scanner_agent_node => undisclosed_tool_call_node => END
builder.add_edge("scanner_agent_node", "undisclosed_tool_call_node")
builder.add_edge("undisclosed_tool_call_node", END)

# 6) response_agent_node => END
builder.add_edge("response_agent_node", END)

# Compile graph
graph = builder.compile()

