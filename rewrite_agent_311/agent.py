import os
import getpass
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.firefox import GeckoDriverManager


##############################
# 1. Environment & Setup
##############################

load_dotenv()

# Prompt for missing environment variables
def _ensure_env_var(var_name: str):
    """Prompt for environment variables if not found."""
    if not os.environ.get(var_name):
        os.environ[var_name] = getpass.getpass(f"{var_name}: ")

for required_var in ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]:
    _ensure_env_var(required_var)

# Optionally set up extra environment variables needed by LangSmith or others.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "browser-agent")


##############################
# 2. State Definitions
##############################

class SimpleGraphState(TypedDict, total=False):
    user_input: str
    rewritten_input: object  # LLM message or string
    needs_browser: bool
    browser_summary: str
    response: str
    error: Optional[str]

class BrowserState(TypedDict, total=False):
    rewritten_input: object
    browser_result: str
    browser_summary: str
    error: Optional[str]


##############################
# 3. Browser Management
##############################

class BrowserManager:
    """Encapsulates Selenium logic in a class for clarity and reusability."""

    @staticmethod
    def create_browser():
        """
        Create and return a Selenium Chrome WebDriver using webdriver-manager.
        In a container with Xvfb, you typically won't run headless
        so you can see the browser through VNC.
        """
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        # Omit headless so we can see the browser in VNC.

        try:
            # Use webdriver_manager to install the matching ChromeDriver
            driver = webdriver.Firefox(
                service=Service(GeckoDriverManager().install()),
                options=options
            )
            path = GeckoDriverManager().install()
            print("Geckodriver path:", path)
            return driver
        except WebDriverException as e:
            print(f"[BrowserManager] Failed to initialize browser: {e}")
            return None

    @staticmethod
    def extract_url_from_query(query: str) -> Optional[str]:
        """Naive approach to extract the first URL from text."""
        tokens = query.split()
        for t in tokens:
            if t.startswith("http://") or t.startswith("https://"):
                return t
        return None

    @staticmethod
    def safe_browser_interaction(query: str) -> tuple[str, Optional[str]]:
        """
        Perform a browser interaction, returning (result_string, error_code).
        error_code is None if all went well, or a short string if an error occurred.
        """
        driver = BrowserManager.create_browser()
        if not driver:
            return "Failed to initialize browser", "browser_init_failed"

        try:
            # Check if the query contains a direct URL
            possible_url = BrowserManager.extract_url_from_query(query)
            if possible_url:
                driver.get(possible_url)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                message = f"Browser opened URL: {driver.current_url}"
            else:
                # If no direct URL, navigate to Google and enter the query
                driver.get("https://www.google.com")
                search_box = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.NAME, "q"))
                )
                search_box.clear()
                search_box.send_keys(query)
                message = f"Opened Google and entered query: {query}"

            # Return without calling driver.quit() so the browser stays open
            return message, None

        except TimeoutException:
            return "Page load timed out", "timeout"
        except WebDriverException as e:
            return f"Browser interaction failed: {str(e)}", "browser_error"
        except Exception as e:
            return f"Unexpected error: {str(e)}", "unknown_error"


##############################
# 4. Graph Logic & Nodes
##############################

class GraphNodes:
    """Collection of node functions for the StateGraph."""

    def __init__(self):
        # Initialize a single instance of your LLM
        self.llm = ChatOpenAI(model="gpt-4")

    def rewrite_input(self, state: SimpleGraphState) -> SimpleGraphState:
        user_input = state["user_input"]
        prompt = f"Rewrite for clarity (include any relevant URLs or search terms):\n\n{user_input}"
        rewritten_msg = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"rewritten_input": rewritten_msg}

    def classify_browser_need(self, state: SimpleGraphState) -> SimpleGraphState:
        """
        Decide whether we need to open a browser.
        You could do a sophisticated check, but here's a basic approach:
        """
        rewritten = state.get("rewritten_input", "")
        content = rewritten.content if hasattr(rewritten, "content") else str(rewritten)

        # If user says "navigate", "visit", "open", "browse" or has HTTP(S) in request, we assume we need a browser.
        triggers = ["navigate", "visit", "open", "browse", "url", "http", "https"]
        state["needs_browser"] = any(t in content.lower() for t in triggers)
        return state

    def browser_interaction(self, state: BrowserState) -> BrowserState:
        """
        Node that actually launches the browser and navigates accordingly.
        """
        rewritten = state["rewritten_input"]
        query = rewritten.content if hasattr(rewritten, "content") else str(rewritten)

        result, error = BrowserManager.safe_browser_interaction(query)
        return {
            "browser_result": result,
            "error": error
        }

    def analyze_browser_result(self, state: BrowserState) -> BrowserState:
        """
        Summarize or interpret what happened in the browser step.
        """
        result = state.get("browser_result", "No result")
        error = state.get("error")

        prompt = f"""
Analyze this browser interaction:
Result: {result}
Error: {error if error else 'None'}

Provide a concise summary and any recommended next steps.
"""
        summary_msg = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"browser_summary": summary_msg}

    def generate_final_response(self, state: SimpleGraphState) -> SimpleGraphState:
        """
        Consolidate everything into a final user-facing response.
        """
        # If we used a browser, we probably have a browser_summary
        # Otherwise, we just rely on the rewritten_input
        content_obj = state.get("browser_summary") or state.get("rewritten_input")
        error = state.get("error")

        content_str = content_obj.content if hasattr(content_obj, "content") else str(content_obj)

        prompt = f"""
Based on the following, provide a final response for the user:

Content: {content_str}
Error: {error if error else 'None'}

Provide next steps or clarifications as needed.
"""
        final_msg = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"response": final_msg}


##############################
# 5. Build the Graph
##############################

def build_graph():
    nodes = GraphNodes()

    # Subgraph for browser
    browser_subgraph = StateGraph(input=BrowserState, output=BrowserState)
    browser_subgraph.add_node("browser_interaction", nodes.browser_interaction)
    browser_subgraph.add_node("analyze_browser", nodes.analyze_browser_result)
    browser_subgraph.add_edge(START, "browser_interaction")
    browser_subgraph.add_edge("browser_interaction", "analyze_browser")
    browser_subgraph.add_edge("analyze_browser", END)
    compiled_browser_subgraph = browser_subgraph.compile()

    # Main graph
    main_graph = StateGraph(input=SimpleGraphState, output=SimpleGraphState)
    main_graph.add_node("rewrite_input", nodes.rewrite_input)
    main_graph.add_node("classify_browser_need", nodes.classify_browser_need)
    main_graph.add_node("browser_subgraph", compiled_browser_subgraph)
    main_graph.add_node("final_response", nodes.generate_final_response)

    # Flow:
    # START -> rewrite_input -> classify_browser_need -> [conditional] -> browser_subgraph -> final_response -> END
    main_graph.add_edge(START, "rewrite_input")
    main_graph.add_edge("rewrite_input", "classify_browser_need")

    # Conditional route
    def route_based_on_need(state: SimpleGraphState) -> str:
        if state.get("needs_browser", False):
            return "browser_subgraph"
        else:
            return "final_response"

    main_graph.add_conditional_edges("classify_browser_need", route_based_on_need)
    main_graph.add_edge("browser_subgraph", "final_response")
    main_graph.add_edge("final_response", END)

    graph = main_graph.compile()
    return graph

graph = build_graph()

##############################
# 6. Run if main
##############################
if __name__ == "__main__":
    graph = build_graph()

    # Example input that references a URL
    initial_state: SimpleGraphState = {
        "user_input": "Please navigate to https://pokeapi.co/api/v2/pokemon/charizard and perform a search."
    }

    final = graph(initial_state)
    print("\n=== Final Response ===")
    # If it's an AIMessage, you can do final["response"].content
    result_obj = final["response"]
    result_str = result_obj.content if hasattr(result_obj, "content") else str(result_obj)
    print(result_str)


