import os
import getpass
import requests
from dotenv import load_dotenv

from pydantic import Field
from typing import Optional

# LangGraph imports
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState

##############################################################################
# 0. Load environment variables
##############################################################################
load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")

# Turn on Langchain/Tracing project if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]


##############################################################################
# 1. Define a Custom State to store relevant data
##############################################################################
class RewriteAgentState(MessagesState):
    """
    Holds the conversation messages plus flags for deciding how to handle
    user queries, plus any stored vector store result.
    """
    scanner_flag: bool = Field(False, description="If set, we must call the scanner tool first.")
    vector_store_answer: Optional[str] = Field(None, description="Stores the result from vector store (if any).")
    user_query: Optional[str] = Field(None, description="The raw user query from the last user message.")
    external_url_needed: bool = Field(False, description="If set, the user wants to query some external URL.")


##############################################################################
# 2. Define Tools (Scanner + Vector Store API)
##############################################################################
def send_to_scanner_tool(scan_url: str) -> str:
    """
    Example 'scanner' tool. In real usage, you'd contact an internal or external
    scanner service. For demonstration, we just do a mock call.
    """
    # Mock external scanner call:
    # Real code could be: 
    # response = requests.post("http://scanner-service/scan", json={"url": scan_url})
    # if response.status_code != 200: 
    #     return f"Scanner call failed: {response.status_code}"
    # ...
    return f"Scanner tool called successfully on: {scan_url}"

def query_vector_store_tool(query: str) -> str:
    """
    Sends the query to a local QA endpoint to retrieve a vector store result.
    This is just a demonstration stub.
    """
    # Example of calling your local QA model at 'localhost:8675/api/models/qa'
    url = "http://localhost:8675/api/models/qa"
    payload = {"query": query}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "<no answer field in JSON>")
        else:
            return f"Error from QA endpoint: {response.status_code}"
    except Exception as e:
        return f"Exception calling QA endpoint: {str(e)}"


##############################################################################
# 3. Instantiate an LLM with the tools bound (if you want function calls)
##############################################################################
# If you want the large GPT-4, just be mindful of usage.
# We'll use a mocked "gpt-4o" name here (replace with your actual deployment name).
llm = ChatOpenAI(model="gpt-4o")

# If you want the LLM to produce function calls to these tools (OpenAI function calling style),
# you can do:
# llm_with_tools = llm.bind_tools([send_to_scanner_tool, query_vector_store_tool])
#
# But for clarity, we'll manually call the tools from the Graph nodes below.


##############################################################################
# 4. Decision Node: Figure out next step
##############################################################################
def decide_node(state: RewriteAgentState):
    """
    Decide whether we must:
      1) Call 'scanner_node' if scanner_flag==True or external_url_needed==True
      2) Else if vector_store_answer is None => go to 'rewrite_and_vector_node'
      3) Else => go to 'final_llm_node'
    """
    if state.scanner_flag or state.external_url_needed:
        return "scanner_node"
    elif state.vector_store_answer is None:
        return "rewrite_and_vector_node"
    else:
        return "final_llm_node"


##############################################################################
# 5. Scanner Node
##############################################################################
def scanner_node(state: RewriteAgentState):
    """
    If we are here, that means we must call the scanner. We'll do so using
    the user query (or a discovered URL from it).
    Then we store the scanner result as a message in the conversation, and continue.
    """
    user_query = state.user_query or "No query provided"
    # For demonstration, let's just pass the entire user query as the 'scan_url'
    result = send_to_scanner_tool(user_query)

    # Add the tool result as an assistant message with role="tool"
    return {
        "messages": [
            {
                "role": "tool",
                "content": result
            }
        ]
    }


##############################################################################
# 6. Rewrite + VectorStore Node
##############################################################################
def rewrite_and_vector_node(state: RewriteAgentState):
    """
    1) Possibly rewrite the user query for clarity or disambiguation. This can be done
       by calling an LLM or your own rewriting logic. 
    2) Then pass the (possibly rewritten) query to the vector store (localhost:8675/api/models/qa).
    3) Store the result in state.vector_store_answer
    """
    user_query = state.user_query or "No query found"

    # 6.1: Attempt rewriting the user query for clarity, using LLM. Example:
    rewrite_prompt = [
        SystemMessage(content="You are a rewriting assistant. Improve clarity of user queries as needed."),
        HumanMessage(content=f"Rewrite for clarity: {user_query}")
    ]
    rewrite_response = llm.invoke(rewrite_prompt)
    refined_query = rewrite_response.content.strip()

    # 6.2: Now call the vector store with refined query
    vs_result = query_vector_store_tool(refined_query)

    # We'll check if we got an error or a real result:
    if vs_result.startswith("Error") or vs_result.startswith("Exception"):
        # If you want to handle the error specially, do so. For now, just store as is:
        new_vs_answer = vs_result
    else:
        new_vs_answer = vs_result  # presumably the text retrieved

    # Update state with the new vector store answer
    state.vector_store_answer = new_vs_answer

    # Return a "tool" role message describing what happened for the conversation flow
    return {
        "messages": [
            {
                "role": "tool",
                "content": f"Vector store result set. Rewritten query: '{refined_query}'\nResult: {new_vs_answer}"
            }
        ]
    }


##############################################################################
# 7. Final LLM Node
##############################################################################
def final_llm_node(state: RewriteAgentState):
    """
    We have user_query and possibly vector_store_answer in state. Let's
    pass them into GPT-4 for a final answer.
    """
    user_query = state.user_query or "No user query found"
    vs_answer = state.vector_store_answer or "No vector store info"

    # Construct a system message that includes the vector store answer
    system = SystemMessage(content="You are an assistant that has some contextual info from a vector store. Use it if relevant.")
    context_msg = f"Vector store context: {vs_answer}\n\nUser's question: {user_query}"
    conversation = [system, HumanMessage(content=context_msg)]

    # Let GPT-4 produce final answer
    final_response = llm.invoke(conversation)
    return {
        "messages": [final_response]
    }


##############################################################################
# 8. Build the Graph
##############################################################################
builder = StateGraph(RewriteAgentState)

# Add nodes
builder.add_node("decide_node", decide_node)
builder.add_node("scanner_node", scanner_node)
builder.add_node("rewrite_and_vector_node", rewrite_and_vector_node)
builder.add_node("final_llm_node", final_llm_node)

# Start => decide_node
builder.add_edge(START, "decide_node")

# decide_node => route to either scanner_node, rewrite_and_vector_node, or final_llm_node
builder.add_conditional_edges("decide_node", decide_node)

# After scanner_node => decide_node again (maybe after scanning, we want to see if we need the vector store next)
builder.add_edge("scanner_node", "decide_node")

# After rewrite_and_vector_node => final_llm_node
builder.add_edge("rewrite_and_vector_node", "final_llm_node")

# final_llm_node => END
builder.add_edge("final_llm_node", END)

# Compile the graph
rewrite_agent_graph = builder.compile()


##############################################################################
# 9. Demonstration of usage
##############################################################################
if __name__ == "__main__":
    # Example user message
    # Suppose the user wants to scan a URL or is just asking a question
    # We'll set 'scanner_flag' to True to simulate a scenario
    initial_state = RewriteAgentState(
        messages=[{"role": "user", "content": "Hi, can you check this URL for threats? http://example.com"}],
        scanner_flag=True,
        user_query="http://example.com",  # pretend the user typed this URL
        external_url_needed=False
    )

    # Run the graph
    final_state = rewrite_agent_graph.run(initial_state)

    # Print the final conversation
    for msg in final_state["messages"]:
        role = msg.get("role", "assistant").upper()
        content = msg["content"]
        print(f"{role}: {content}")


