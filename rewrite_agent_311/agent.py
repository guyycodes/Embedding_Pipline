#agent.py
import os
import getpass
from typing import TypedDict, Optional
import logging
import subprocess
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    @classmethod
    def initialize_display(cls):
        """Initialize display settings."""
        # Explicitly set display if not set
        if not os.environ.get('DISPLAY'):
            logger.debug("DISPLAY not set, setting to :0")
            os.environ['DISPLAY'] = ':0'
        
        display = os.environ.get('DISPLAY')
        logger.debug(f"Using X display: {display}")
        
        # Verify X server is running
        try:
            subprocess.check_call(['xdpyinfo'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug("Successfully connected to X server")
        except subprocess.CalledProcessError:
            logger.error("Could not connect to X server")
            # Check if Xvfb is actually running
            try:
                xvfb_check = subprocess.check_output(['ps', 'aux']).decode()
                if 'Xvfb' in xvfb_check:
                    logger.debug("Xvfb is running but xdpyinfo failed. This might be a permissions issue.")
                else:
                    logger.debug("No Xvfb process found.")
            except Exception:
                pass
            raise RuntimeError("X server not running or accessible")

    @staticmethod
    def create_browser():
        """
        Create and return a Selenium Firefox WebDriver using webdriver-manager.
        This version includes verbose logging to help diagnose startup issues.
        """
        logger.debug("Starting create_browser()")
        
        # Initialize display first
        BrowserManager.initialize_display()
        
        logger.debug(f"Current DISPLAY: {os.environ.get('DISPLAY')}")
        
        # Create log directory if it doesn't exist
        if not os.path.exists('/tmp'):
            os.makedirs('/tmp', exist_ok=True)
        
        os.environ['GECKO_LOG'] = 'trace'

        options = Options()
        # Start with minimal options
        options.add_argument('--no-sandbox')
        # options.add_argument('--disable-dev-shm-usage')
        
        # Only set critical preferences
        # options.set_preference('browser.cache.disk.enable', False)
        # options.set_preference('dom.ipc.processCount', 1)
        # options.add_argument("--headless")
        # options.add_argument('--no-sandbox')
        # options.add_argument('--disable-dev-shm-usage')
        # options.add_argument('--profile')
        # options.add_argument('/root/.mozilla/firefox/*.selenium')
        # options.set_preference('browser.cache.disk.enable', False)
        # options.set_preference('browser.cache.memory.enable', False)
        # options.set_preference('browser.cache.offline.enable', False)
        # options.set_preference('network.http.use-cache', False)
        # # Omit headless so we can see the browser through VNC.
        # options.set_preference('dom.ipc.processCount', 1)
        # options.set_preference('javascript.options.mem.max', 512)
        #     # Add these memory-related preferences
        # options.set_preference('browser.sessionhistory.max_entries', 10)
        # options.set_preference('browser.sessionhistory.max_total_viewers', 4)
        # options.set_preference('browser.tabs.remote.warmup.enabled', False)

        try:
            # Install and cache the geckodriver, and log its path
            driver_path = GeckoDriverManager(version="v0.35.0").install()
            logger.debug("Geckodriver path: %s", driver_path)
            
            # Pass a log file path for the driver service
            service = Service(driver_path)
            
            # Initialize the Firefox driver
            driver = webdriver.Firefox(service=service, options=options)
            logger.debug("Firefox driver successfully started.")
            return driver

        except WebDriverException as e:
            logger.error("Failed to initialize browser (WebDriverException): %s", e, exc_info=True)
            return None
        except Exception as e:
            logger.error("Unexpected error during browser initialization: %s", e, exc_info=True)
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
        logger.debug("Starting safe_browser_interaction() with query: %s", query)
        driver = BrowserManager.create_browser()
        if not driver:
            logger.error("Browser initialization failed.")
            return "Failed to initialize browser", "browser_init_failed"

        try:
            possible_url = BrowserManager.extract_url_from_query(query)
            if possible_url:
                logger.debug("Navigating to URL: %s", possible_url)
                driver.get(possible_url)

                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                message = f"Browser opened URL: {driver.current_url}"
            else:
                logger.debug("No URL found in query. Navigating to Google.")
                driver.get("https://www.google.com")


                search_box = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.NAME, "q"))
                )
                search_box.clear()
                search_box.send_keys(query)
                message = f"Opened Google and entered query: {query}"

            # Optionally, you might eventually quit the driver.
            logger.debug("Browser interaction succeeded with message: %s", message)
            return message, None

        except Exception as e:
            logger.error("Exception during browser interaction: %s", e, exc_info=True)
            if "timeout" in str(e).lower():
                return "Page load timed out", "timeout"
            return f"Unexpected error: {str(e)}", "unknown_error"


##############################
# 4. Graph Logic & Nodes
##############################

class GraphNodes:
    """Collection of node functions for the StateGraph."""

    def __init__(self):
        # Initialize a single instance of your LLM
        self.llm = ChatOpenAI(model="gpt-4.5-preview")
        self.cheapllm = ChatOpenAI(model="gpt-4o")

    def rewrite_input(self, state: SimpleGraphState) -> SimpleGraphState:
        user_input = state["user_input"]
        prompt = f"The user will ask for help with problems and may provide code, using software engineering terminology please do the following: Rewrite the users text query about the code more clearly, ensure all the code apart of the query gets included with your optimized response:\n\n{user_input}"
                # prompt = f"""You are a software engineering assistant.

        #         When analyzing the user's code problem:
        #         1. Interpret their query using precise technical terminology
        #         2. Rewrite their question to be more specific and actionable
        #         3. Include ALL original code snippets in your response without modification

        #         Focus on clarifying the technical intent while preserving every line of code.

        #         User query:
        #         {user_input}"""
        # prompt = f"""You are an assistant, the user may prompt you and also supply cntent they are needing help with.

        #         Identify the users prompt and distinguish it from the users supplied content, then:
        #         - Identify what the user wants
        #         - If there is any mis-spelling or lack of clarity to the users prompt, rewrite and optimize the users prompt.
        #         - Preserve the original intent
        #         - Preserve the orginal content
        #         - When finished, your response will be passed to greater a.i model, ensure you are not adding in garbage information for the next a.i model.

        #         If the query is already clear:
        #         - Pass it through unchanged

        #         IMPORTANT: Always include ALL original content/text the user provided in your response.

        #         User query:
        #         {user_input}"""
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
            Based on the following, provide a proper response to the user:

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
    # browser_subgraph = StateGraph(input=BrowserState, output=BrowserState)
    # browser_subgraph.add_node("browser_interaction", nodes.browser_interaction)
    # browser_subgraph.add_node("analyze_browser", nodes.analyze_browser_result)
    # browser_subgraph.add_edge(START, "browser_interaction")
    # browser_subgraph.add_edge("browser_interaction", "analyze_browser")
    # browser_subgraph.add_edge("analyze_browser", END)
    # compiled_browser_subgraph = browser_subgraph.compile()

    # Main graph
    main_graph = StateGraph(input=SimpleGraphState, output=SimpleGraphState)
    main_graph.add_node("rewrite_input", nodes.rewrite_input)
    main_graph.add_node("classify_browser_need", nodes.classify_browser_need)
    # main_graph.add_node("browser_subgraph", compiled_browser_subgraph)
    main_graph.add_node("final_response", nodes.generate_final_response)

    # Flow:
    # START -> rewrite_input -> classify_browser_need -> [conditional] -> browser_subgraph -> final_response -> END
    main_graph.add_edge(START, "rewrite_input")
    main_graph.add_edge("rewrite_input", "classify_browser_need")

    # Conditional route
    # def route_based_on_need(state: SimpleGraphState) -> str:
    #     if state.get("needs_browser", False):
    #         return "browser_subgraph"
    #     else:
    #         return "final_response"
    
    def route_based_on_need(state: SimpleGraphState) -> str:
        # if state.get("needs_browser", False):
        #     return "browser_subgraph"
        # else:
        return "final_response"

    main_graph.add_conditional_edges("classify_browser_need", route_based_on_need)
    # main_graph.add_edge("browser_subgraph", "final_response")
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
        "user_input": "Please navigate to https://duckduckgo.com and perform a search for toliet paper."
    }

    final = graph(initial_state)
    print("\n=== Final Response ===")
    # If it's an AIMessage, you can do final["response"].content
    result_obj = final["response"]
    result_str = result_obj.content if hasattr(result_obj, "content") else str(result_obj)
    print(result_str)


