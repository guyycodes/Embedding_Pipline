import os
import getpass
from dotenv import load_dotenv
import requests
from typing import Tuple

##############################
# LangGraph & LangChain Imports
##############################
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState

##############################
# 0. Environment & Setup 
# Objective : Rewrite → (conditional) → Scanner or Vector Store → Response Agent.
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
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "")

##############################
# 1. State Definition
##############################
class RewriteFlowState(MessagesState):
    """
    A custom state to keep track of all necessary data for our flow.
    """
    # The raw user query from the last user message
    user_query: str = ""
    
    # Flag indicating if external data is needed (i.e. route to Scanner)
    requires_external_data: bool = False

    # Storage for vector store results (if any)
    vector_store_answer: str = ""

    # Tracks the status code returned by vector store (if relevant)
    vs_status_code: int = 0

##############################
# 2. LLM Definitions
##############################

# The "Rewrite Agent" is a GPT-4 model that:
#   - Analyzes user query
#   - Decides if external data is needed
#   - Produces an optimized query

rewrite_model = ChatOpenAI(
    model="gpt-4",  # Using GPT-4 for rewriting
    temperature=0
)

# The "Response Agent" is a GPT-4o model that:
#   - Receives original user query & vector store results
#   - Produces final response or clarifying question

response_model = ChatOpenAI(
    model="gpt-4o",  # Using a hypothetical "gpt-4o" endpoint
    temperature=0
)

# The "Scanner Agent" is a GPT-o3-Mini model (not implemented yet).
#   - Placeholder only, for future external data processing.

# scanner_model = ChatOpenAI(
#     model="gpt-o3-mini",  # Future placeholder
#     temperature=0
# )


##############################
# 3. Tools
##############################
def vector_store_tool(optimized_query: str) -> Tuple[str, int]:
    """
    Call the vector store endpoint at localhost:8675/api/models/qa
        Args:
        optimized_query: str
    Returns tuple: (answer_string, status_code).
    """
    url = "http://localhost:8675/api/models/qa"
    payload = {"query": optimized_query}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            json_data = response.json()
            # We assume there's an "answer" field in the JSON
            answer = json_data.get("answer", "")
            return answer, 200
        else:
            return f"Vector store lookup failed. Code: {response.status_code}", response.status_code
    except Exception as e:
        return f"Exception calling QA endpoint: {str(e)}", 500


##############################
# 4. Node Implementations
##############################

def rewrite_agent_node(state: RewriteFlowState):
    """
    Node for the Rewrite Agent:
      * Takes user_query from state
      * Analyzes whether external data is needed
      * Produces an optimized query for vector store
      * Sets `requires_external_data` in state accordingly
    """
    user_query = state.user_query or "No user query given."

    # Craft a minimal prompt to the rewrite model
    # The rewrite model should:
    #  1) Check if the user query needs external data
    #  2) Rewrite it for clarity
    #  3) Output both the route decision & the optimized query
    rewrite_prompt = [
        SystemMessage(content="""You are the Rewrite Agent.
            1. Determine whether user query requires external scanning (set requires_external_data=True if so).
            2. If no external data needed, set requires_external_data=False.
            3. Produce an 'optimized' query for internal usage.
            4. Output JSON with structure: 
    {
      "requires_external_data": true/false,
      "optimized_query": "..."
    }"""),
        HumanMessage(content=f"User query: {user_query}")
    ]
    
    # Get the LLM's "raw" string response
    raw_response = rewrite_model.invoke(rewrite_prompt)
    
    # For a robust approach, parse JSON. For simplicity, do naive parse:
    import json
    try:
        # Attempt to parse the response as JSON
        parsed = json.loads(raw_response)
        requires_external_data = parsed.get("requires_external_data", False)
        optimized_query = parsed.get("optimized_query", user_query)
    except:
        # Fallback if parsing fails, do a naive approach
        requires_external_data = False
        optimized_query = user_query
    
    # Update state
    state.requires_external_data = requires_external_data
    state.user_query = user_query  # keep original
    # We'll store the *optimized_query* in a message so the next node can see it
    # or we can store in state for direct usage
    # Let's store as an "assistant" message for clarity or store directly in state
    # We'll store as an "assistant" message so you see it in final logs
    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"Rewrite done. requires_external_data={requires_external_data}. optimized_query={optimized_query}"
            }
        ],
        "optimized_query": optimized_query
    }


def scanner_agent_node(state: RewriteFlowState, node_result):
    """
    Placeholder for future Scanner Agent logic (GPT-o3-Mini).
    For now, we just comment out the actual logic. We'll simulate a pass-through.
    
    In the future, we would do something like:
      1) Summarize or transform data externally
      2) Return some result to store in the conversation or state
    """
    # In a real scenario, we might do something like:
    #
    # scanner_prompt = [
    #   SystemMessage(content="You are a Scanner Agent..."),
    #   ...
    # ]
    # response = scanner_model.invoke(scanner_prompt)
    #
    # placeholder_result = response.content
    #
    # For now, we simply produce a placeholder message:
    placeholder_result = (
        "Scanner Agent Placeholder - no action taken. (Future GPT-o3-Mini implementation.)"
    )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": placeholder_result
            }
        ]
    }


def vector_store_node(state: RewriteFlowState, node_result):
    """
    Node for calling the Vector Store with an optimized query.
    The 'optimized_query' was returned from rewrite_agent_node in node_result["optimized_query"].
    - We pass that to the vector_store_tool
    - We store result in vector_store_answer if success
    - or note that there's no results if not success
    """
    # Extract the optimized query from the previous node result
    optimized_query = node_result.get("optimized_query", state.user_query)
    
    answer, status_code = vector_store_tool(optimized_query)

    state.vs_status_code = status_code
    # If successful, summarize the answer; otherwise store empty or error text
    if status_code == 200 and answer:
        # Summarize the "top insights" to reduce token usage
        summary_prompt = [
            SystemMessage(content="You are a Summarizer. Provide only the top insights from the text below, in concise form."),
            HumanMessage(content=answer)
        ]
        try:
            summary = rewrite_model.invoke(summary_prompt)
            # Trim or clean up as needed
            summary = summary.strip()
            state.vector_store_answer = summary
        except Exception as e:
            # Fallback: if summarization fails, store partial
            state.vector_store_answer = answer[:500]  # or some short fallback
    else:
        state.vector_store_answer = ""

    return {
        "messages": [
            {
                "role": "tool",
                "content": f"Vector store call returned status={status_code}. Output: {answer}"
            }
        ]
    }


def response_agent_node(state: RewriteFlowState):
    """
    Final Response node.
    We have:
      - user_query (original)
      - vector_store_answer (summarized top insights, if any)
    We'll pass them to GPT-4o to generate a final reply.
    If vector_store_answer is empty, produce a clarifying question or note no data found.
    """
    user_query = state.user_query
    vs_answer = state.vector_store_answer or ""

    # We'll create a system prompt:
    sys_prompt = """You are the final Response Agent. 
You receive the user's original query and any available vector store data. 
If there's NO vector store data, ask clarifying questions or mention no relevant data found.
If there IS vector store data, incorporate it into your final answer.
Do not tell the user you retreived data from anywhere.
"""
    # Then a user-like message describing the scenario
    content_msg = f"User query: {user_query}\nVector Store Data: {vs_answer}"

    # We'll call the response_model with these messages:
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=content_msg)
    ]
    final_answer = response_model.invoke(messages)

    return {
        "messages": [final_answer]
    }


##############################
# 5. Routing Logic
##############################
def route_from_rewrite(state: RewriteFlowState, node_result):
    """
    The rewrite node sets requires_external_data in state. 
    This decides whether we go to the Scanner Agent node or Vector Store node next.
    """
    if state.requires_external_data:
        return "scanner_agent_node"
    else:
        return "vector_store_node"

def route_after_scanner(state: RewriteFlowState, node_result):
    """
    After the scanner is done, let's assume we always proceed to vector store next.
    (One could implement a more complex logic if needed.)
    """
    return "vector_store_node"

def route_after_vector(state: RewriteFlowState, node_result):
    """
    After vector store, we go to the final response agent.
    """
    return "response_agent_node"


##############################
# 6. Building the Graph
##############################
builder = StateGraph(RewriteFlowState)

# Add nodes
builder.add_node("rewrite_agent_node", rewrite_agent_node)
builder.add_node("scanner_agent_node", scanner_agent_node)
builder.add_node("vector_store_node", vector_store_node)
builder.add_node("response_agent_node", response_agent_node)

# Start -> rewrite
builder.add_edge(START, "rewrite_agent_node")

# rewrite -> conditional -> (scanner OR vector_store)
builder.add_conditional_edges("rewrite_agent_node", route_from_rewrite)

# scanner -> vector store
builder.add_conditional_edges("scanner_agent_node", route_after_scanner)

# vector store -> final response
builder.add_conditional_edges("vector_store_node", route_after_vector)

# final response -> end
builder.add_edge("response_agent_node", END)

# Compile
graph = builder.compile()

##############################
# 7. Example Usage
##############################
if __name__ == "__main__":
    # Example user query
    user_input = "What's the best approach to automatically detect anomalies in a large dataset? Possibly need an external data check."

    # Create initial state with user's message
    init_state = RewriteFlowState(
        messages=[{"role": "user", "content": user_input}],
        user_query=user_input
    )

    # Run the graph
    final_state = graph.run(init_state)

    # Print final conversation messages
    for msg in final_state["messages"]:
        role = msg.get("role", "assistant").upper()
        content = msg["content"]
        print(f"{role}: {content}")
