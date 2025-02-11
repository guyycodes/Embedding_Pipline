import os, getpass
from dotenv import load_dotenv
import requests
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from pydantic import Field

load_dotenv()

# If running interactively, prompt for environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")

# Turn on Langchain/Tracing project if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]


##############################################################################
# 1. Define the Tool for calling external API and Docker container
##############################################################################
def call_regulatory_api(api_endpoint: str) -> str:
    """
    Call an external API to get regulatory information.
    If successful (200), call a Docker container endpoint with the fetched data.
    If the Docker call is also 200, return success message. Otherwise, end.
    """
    try:
        # 1. Fetch data from external API
        ext_response = requests.get(api_endpoint)
        if ext_response.status_code != 200:
            return f"Tool call ended: External API call failed with code {ext_response.status_code}"
        
        data = ext_response.json()  # regulatory info data

        # 2. Pass the data to your Docker container (assuming it’s running locally)
        docker_endpoint = "http://localhost:5000/my-docker-endpoint"
        docker_response = requests.post(docker_endpoint, json=data)
        if docker_response.status_code != 200:
            return f"Tool call ended: Docker call failed with code {docker_response.status_code}"

        return "Tool call success: Regulatory data retrieved and Docker updated."
    
    except Exception as e:
        return f"Tool call ended: Error occurred -> {str(e)}"


##############################################################################
# 2. Bind the LLM to this single tool
##############################################################################
llm = ChatOpenAI(model="gpt-4")  # or "o3-mini", or any model ID you prefer
llm_with_tools = llm.bind_tools([call_regulatory_api])

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
##############################################################################
# 3. Define a Custom State to track depth
##############################################################################
class CustomMessagesState(MessagesState):
    iteration_count: int = Field(0, description="How many times we've looped.")
    max_depth: int = Field(3, description="Maximum number of loops allowed.")


##############################################################################
# 4. Assistant Node
##############################################################################
def assistant_node(state: CustomMessagesState):
    """
    Each time we invoke the assistant, increment iteration_count
    and stop if it exceeds max_depth. Otherwise, let the LLM respond.
    """
    state.iteration_count += 1
    
    # If we've exceeded the max depth, end immediately
    if state.iteration_count > state.max_depth:
        return {
            "messages": [
                {"role": "assistant", "content": "Reached max depth. Stopping now."}
            ]
        }
    
    # If within depth, let the LLM respond
    # Note: no special system prompt; you can add your own instructions or remove entirely
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


##############################################################################
# 5. Tool Node
##############################################################################
def tool_node(state: MessagesState):
    """
    Execute the single tool call. The LLM's last message should have
    the arguments if it decided to call `call_regulatory_api`.
    """
    last_msg = state["messages"][-1]
    # The LLM tool format typically returns something like:
    # {
    #   "name": "call_regulatory_api",
    #   "arguments": { "api_endpoint": "..." }
    # }
    arguments = last_msg.get("arguments", {})
    endpoint = arguments.get("api_endpoint", "")

    result_str = call_regulatory_api(endpoint)
    return {"messages": [
        {"role": "tool", "content": result_str}
    ]}


##############################################################################
# 6. Custom Condition: Decide if we want to call the tool or end
##############################################################################
def api_condition(state: MessagesState, node_result):
    """
    Inspect the last node_result to see if the LLM is requesting
    a tool call to 'call_regulatory_api'. If yes, route to 'tool_node',
    else route to 'END'.
    """
    # node_result is typically {"messages": [ message1, message2, ... ]}.
    # The latest message is node_result["messages"][-1].
    last_msg = node_result["messages"][-1]

    # The LLM (in OpenAI function-call mode) might produce something like:
    # { "role": "assistant", "content": ... } or
    # { "role": "assistant", "name": "call_regulatory_api", "arguments": {...} }
    tool_name = last_msg.get("name", None)

    if tool_name == "call_regulatory_api":
        return "tool_node"
    else:
        return "END"


##############################################################################
# 7. Build the Graph
##############################################################################
builder = StateGraph(CustomMessagesState)

builder.add_node("assistant_node", assistant_node)
builder.add_node("tool_node", tool_node)

# Start -> assistant node
builder.add_edge(START, "assistant_node")

# assistant_node => condition => either call "tool_node" or END
builder.add_conditional_edges("assistant_node", api_condition)

# If the tool is called, once it's done, we go back to "assistant_node" 
# for another round (the LLM might want to parse the new data and respond).
builder.add_edge("tool_node", "assistant_node")

# Compile the graph
graph = builder.compile()


##############################################################################
# 8. Run a Test
##############################################################################
if __name__ == "__main__":
    # Example user message prompting the model to get regulatory data
    initial_state = CustomMessagesState(messages=[
        {"role": "user", "content": "Please retrieve the latest regulatory data from https://example.com/reg-updates and update our container."}
    ])

    final_state = graph.run(initial_state)

    # Print final messages
    for msg in final_state["messages"]:
        print(f"{msg.get('role', 'unknown').upper()}: {msg['content']}")
