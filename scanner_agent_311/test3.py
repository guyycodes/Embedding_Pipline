import os
import json
import hashlib
import getpass
from datetime import datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from minio import Minio
from pydantic import BaseModel, Field

# LangChain / LLM modules (example placeholders)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

#####################################################
# 1. Load environment (API keys, MinIO credentials) #
#####################################################
load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
# MinIO config (sample; set to your actual environment)
_set_env("MINIO_ENDPOINT")
_set_env("MINIO_ACCESS_KEY")
_set_env("MINIO_SECRET_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]

########################################
# 3. Regulation change model & monitor #
########################################
class RegulationChange(BaseModel):
    """Model to track regulation changes"""
    regulation_id: str
    content_hash: str
    last_updated: datetime
    details: Dict

class FDAMonitor:
    """
    Responsible for:
      - Checking MinIO for existing regulation data.
      - Fetching updated data from an FDA endpoint.
      - Detecting changes (via content_hash).
      - Storing new or updated records back to MinIO.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Example bucket name; update to suit your environment
        self.bucket_name = "fda-regulations"
        
        # Instantiate Minio client
        self.minio_client = Minio(
            os.environ["MINIO_ENDPOINT"],
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            secure=False  # or True, depending on your setup
        )
        
        # Ensure bucket exists
        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(self.bucket_name)

    def get_content_hash(self, content: Dict) -> str:
        """Generate a reproducible hash for the regulation content."""
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def fetch_stored_regulations(self) -> Dict[str, RegulationChange]:
        """
        Retrieve the existing regulations from MinIO.
        Returns an empty dict if none are found.
        """
        object_name = "regulations.json"
        try:
            response = self.minio_client.get_object(self.bucket_name, object_name)
            data = json.loads(response.read().decode())
            response.close()
            response.release_conn()
            # Convert JSON to RegulationChange objects
            return {k: RegulationChange(**v) for k, v in data.items()}
        except Exception:
            # If the object doesn't exist or fetch fails
            return {}

    def store_regulations(self, known_regulations: Dict[str, RegulationChange]):
        """
        Store the current set of known regulations in MinIO.
        Overwrites the 'regulations.json' object.
        """
        object_name = "regulations.json"
        as_json_str = json.dumps(
            {k: v.dict() for k, v in known_regulations.items()},
            indent=2
        )
        # Upload to MinIO
        self.minio_client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=as_json_str.encode(),
            length=len(as_json_str)
        )

    async def check_fda_updates(self) -> List[RegulationChange]:
        """
        Check FDA API for updates.
        If none exist in MinIO, store entire set as new.
        Otherwise compare and store changes.
        
        Returns a list of changed (new or updated) regulations.
        """
        # (1) Load known regulations from MinIO
        known_regulations = self.fetch_stored_regulations()

        # (2) Pull current regulations from FDA
        # Example FDA API endpoint (replace with real)
        api_url = "https://api.fda.gov/drug/regulation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            current_regulations = response.json()  # Example structure
        except Exception as e:
            print(f"Error fetching FDA data: {e}")
            return []

        # (3) Compare new data vs. known data
        changes = []
        for reg_id, content in current_regulations.items():
            current_hash = self.get_content_hash(content)
            
            if reg_id not in known_regulations:
                # New regulation
                reg_change = RegulationChange(
                    regulation_id=reg_id,
                    content_hash=current_hash,
                    last_updated=datetime.now(),
                    details=content
                )
                known_regulations[reg_id] = reg_change
                changes.append(reg_change)
            else:
                # Existing regulation - check for changes
                if known_regulations[reg_id].content_hash != current_hash:
                    reg_change = RegulationChange(
                        regulation_id=reg_id,
                        content_hash=current_hash,
                        last_updated=datetime.now(),
                        details=content
                    )
                    known_regulations[reg_id] = reg_change
                    changes.append(reg_change)

        # (4) If anything changed, re-store to MinIO
        if changes:
            self.store_regulations(known_regulations)

        return changes

########################################
# 4. A tool function for your LLM flow #
########################################
import asyncio

async def check_regulations(state: "CustomMessagesState"):
    """
    The tool that will be called by the LLM or the scheduler
    to check for regulation updates.
    """
    # Instantiate FDAMonitor with your FDA API key
    fda_monitor = FDAMonitor(api_key=os.environ.get("FDA_API_KEY", ""))
    
    # Check for updates
    changes = await fda_monitor.check_fda_updates()
    
    if changes:
        # Return metadata about changes
        return {
            "message": f"Found {len(changes)} regulation changes",
            "changes": [chg.dict() for chg in changes]
        }
    return {"message": "No changes found"}

# Extend your tool list with check_regulations
tools = [check_regulations]

##########################################
# 5. Define your system + scanning logic #
##########################################

# System message guiding the overall LLM behavior
sys_msg = SystemMessage(content="""You are a helpful assistant tasked with monitoring FDA regulations.
When asked about regulations, you may call the 'check_regulations' tool to see if there are updates.
If changes are found, provide a summary of those changes.
""")

class CustomMessagesState(MessagesState):
    iteration_count: int = Field(0, description="How many times we've looped.")
    max_depth: int = Field(3, description="Maximum number of loops allowed.")

# Wrap your LLM with tool binding
llm = ChatOpenAI(model="o3-mini")
llm_with_tools = llm.bind_tools(tools)

########################################
# 6. The updated assistant node method #
########################################
async def assistant(state: CustomMessagesState):
    """
    Main "assistant" function that checks for regulation-related queries
    and possibly calls the check_regulations tool.
    """
    state.iteration_count += 1
    
    if state.iteration_count > state.max_depth:
        return {"messages": ["Reached max depth. No more tool calls allowed."]}
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Simple check to see if user wants regulation data
    if "regulation" in last_message.lower():
        # Automatically check regulations
        regulation_check_result = await check_regulations(state)
        
        if regulation_check_result.get("changes"):
            context = (
                f"Found regulation changes:\n"
                f"{json.dumps(regulation_check_result['changes'], indent=2)}"
            )
        else:
            context = "No regulation changes found."
        
        # Add context to system message
        temp_sys_msg = SystemMessage(content=sys_msg.content + f"\n\nContext: {context}")
        response = llm_with_tools.invoke([temp_sys_msg] + state["messages"])
    else:
        # If not a regulation-related query, just proceed normally
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
    
    return {"messages": [response]}

###################################################
# 7. Set up an APScheduler job to run periodically #
###################################################
scheduler = AsyncIOScheduler()

# This runs daily, you can adjust the interval to hours=1 or minutes=30, etc.
@scheduler.scheduled_job('interval', hours=24)
def scheduled_regulation_check():
    """
    This job runs daily to check the regulations automatically.
    Because 'check_regulations' is async, we run it via asyncio.
    """
    loop = asyncio.get_event_loop()
    state = CustomMessagesState(messages=[], iteration_count=0, max_depth=1)
    result = loop.run_until_complete(check_regulations(state))
    print(f"[Scheduled Job] {result}")

# Start the scheduler
scheduler.start()

#########################################
# 8. Graph building (for advanced flows) #
#########################################
builder = StateGraph(CustomMessagesState)
builder.add_node("scanner_assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# If the LLM wants to call a tool, route to the tools node, else end
builder.add_edge(START, "scanner_assistant")
builder.add_conditional_edges("scanner_assistant", tools_condition)
builder.add_edge("tools", "scanner_assistant")

graph = builder.compile()

################################################
# USAGE EXAMPLE (trigger from your main script) #
################################################
if __name__ == "__main__":
    # Example: manual invocation of the graph
    # (In a real scenario, you'd tie this to your web server or command line)
    
    initial_state = CustomMessagesState(messages=[
        HumanMessage(content="Can you check for any new regulation updates?")
    ])
    
    # Run the graph
    output = asyncio.run(graph.run(initial_state))
    print("Final output:", output)
