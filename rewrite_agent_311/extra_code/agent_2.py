import os, getpass, re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
# from langchain_community.tools import TavilySearchResults  # Retained for legacy reasons

# Import Selenium and webdriver-manager components.
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

##############################
# 0. Environment & Setup 
##############################

def _set_env(var: str):
    """Prompt for environment variables if not found."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
# _set_env("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "")
os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "")

# (Optional) Print out env vars (be cautious with printing secret keys in production)
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
print("LANGSMITH_API_KEY:", os.environ.get("LANGSMITH_API_KEY"))
print("TAVILY_API_KEY:", os.environ.get("TAVILY_API_KEY"))

# ================================
# Define state schemas
# ================================
class SimpleGraphState(TypedDict, total=False):
    user_input: str
    rewritten_input: object  # could be an AIMessage or str
    needs_external: bool
    scanner_summary: str
    response: str

class ScannerState(TypedDict, total=False):
    rewritten_input: object  # could be an AIMessage or str
    scanner_result: str
    scanner_summary: str

# ================================
# Initialize the LLM
# ================================
llm = ChatOpenAI(model="gpt-4")

# ================================
# Main Graph Nodes
# ================================

# Node: Rewrite user input for clarity.
def rewrite_user_input(state: SimpleGraphState) -> SimpleGraphState:
    user_input = state["user_input"]
    prompt = f"Please rewrite the following input for clarity:\n\n{user_input}"
    rewritten = llm.invoke([{"role": "user", "content": prompt}])
    return {"rewritten_input": rewritten}

# Node: Classify whether the rewritten input requires external resource access.
def classify_external_node(state: SimpleGraphState) -> SimpleGraphState:
    rewritten = state.get("rewritten_input", "")
    if hasattr(rewritten, "content"):
        rewritten_text = rewritten.content
    else:
        rewritten_text = str(rewritten)
    
    # Simple keyword check.
    trigger_keywords = ["api", "http", "https", "pokeapi", "json", "endpoint", "documentation", "rate limit"]
    if any(kw in rewritten_text.lower() for kw in trigger_keywords):
        state["needs_external"] = True
        return state

    # Otherwise, ask the LLM to decide.
    classification_prompt = f"""
    Given the following rewritten user query, does it appear to require external resource access 
    (for example, making an API call or a web lookup)? 
    Answer with a single word: YES or NO.
    
    Query: {rewritten_text}
    """
    classification_response = llm.invoke([{"role": "user", "content": classification_prompt}])
    answer = classification_response.content.strip().lower()
    state["needs_external"] = answer.startswith("yes")
    return state

# Node: Generate a final response (chain-of-thought) using clarified input.
def respond_to_input(state: SimpleGraphState) -> SimpleGraphState:
    clarified = state.get("scanner_summary", state["rewritten_input"])
    if hasattr(clarified, "content"):
        clarified_text = clarified.content
    else:
        clarified_text = str(clarified)
    
    prompt = f"""
    Please carefully consider the following clarified input.
    Provide your chain-of-thought by explaining your reasoning step by step.
    Even if you already know the answer, deliberately take at least 5 seconds to reason before giving your final answer.
    
    Clarified Input: {clarified_text}
    
    Your final answer should be clearly indicated after your chain-of-thought.
    """
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"response": response}

# ================================
# Subgraph: External Resource Access using Selenium (with webdriver-manager)
# ================================
# This function launches a full browser, navigates to Google,
# and places the query text into the search bar.
def selenium_search(query: str) -> str:
    # Use webdriver-manager to automatically manage the ChromeDriver.
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # If you want to run headless inside the container, uncomment:
    # options.add_argument('--headless')

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    # Navigate to Google (you can change this URL if desired).
    driver.get("https://www.google.com")
    
    # Wait until the search box (input element with name 'q') is present.
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    
    # Clear the search box and enter the query.
    search_box.clear()
    search_box.send_keys(query)
    
    # Do NOT submit the query automatically so the browser remains open.
    result_message = f"Browser launched at {driver.current_url} with query entered: {query}"
    
    # Note: We intentionally do not call driver.quit() so that you can see the browser window.
    return result_message

# Updated scanner_node using Selenium.
def scanner_node(state: ScannerState) -> ScannerState:
    rewritten_input = state["rewritten_input"]
    if hasattr(rewritten_input, "content"):
        query_text = rewritten_input.content
    else:
        query_text = str(rewritten_input)
    
    result = selenium_search(query_text)
    return {"scanner_result": result}

# Node: undisclosed_agent that reasons about the tool’s result and summarizes it.
def undisclosed_agent(state: ScannerState) -> ScannerState:
    scanner_result = state["scanner_result"]
    prompt = f"""
    Please review the following external data:
    {scanner_result}
    
    Now, provide a detailed summary of this data. Even if you immediately understand it,
    deliberately take at least 5 seconds to reason through the content and explain your chain-of-thought before giving the final summary.
    """
    summary = llm.invoke([{"role": "user", "content": prompt}])
    return {"scanner_summary": summary}

# Build the subgraph for scanner functionality.
scanner_builder = StateGraph(input=ScannerState, output=ScannerState)
scanner_builder.add_node("scanner_node", scanner_node)
scanner_builder.add_node("undisclosed_agent", undisclosed_agent)
scanner_builder.add_edge(START, "scanner_node")
scanner_builder.add_edge("scanner_node", "undisclosed_agent")
scanner_builder.add_edge("undisclosed_agent", END)
compiled_scanner_subgraph = scanner_builder.compile()

# ================================
# Conditional Routing Function
# ================================
def routing_from_classification(state: SimpleGraphState) -> str:
    if state.get("needs_external", False):
        return "scanner_subgraph"
    else:
        return "respond"

# ================================
# Build the Main Graph
# ================================
graph_builder = StateGraph(input=SimpleGraphState, output=SimpleGraphState)
graph_builder.add_node("rewrite", rewrite_user_input)
graph_builder.add_node("classify_external", classify_external_node)
graph_builder.add_node("respond", respond_to_input)
graph_builder.add_node("scanner_subgraph", compiled_scanner_subgraph)

# Define the flow:
# START -> rewrite -> classify_external -> (conditional branch) -> scanner_subgraph (if needed) -> respond -> END
graph_builder.add_edge(START, "rewrite")
graph_builder.add_edge("rewrite", "classify_external")
graph_builder.add_conditional_edges("classify_external", routing_from_classification)
graph_builder.add_edge("scanner_subgraph", "respond")
graph_builder.add_edge("respond", END)

graph = graph_builder.compile()

# ================================
# Example Usage
# ================================
if __name__ == "__main__":
    # Example input that should trigger external access (Selenium search).
    initial_state: SimpleGraphState = {
        "user_input": (
            "Please navigate to https://pokeapi.co/api/v2/pokemon/charizard and perform a search."
        )
    }
    final_state = graph(initial_state)
    print("Final Response:\n", final_state["response"])



# "Please navigate to https://pokeapi.co/api/v2/pokemon/charizard and perform a search."
