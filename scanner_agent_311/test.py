import os
import json
import hashlib
import getpass
import requests
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from minio import Minio
from pydantic import BaseModel, Field

# Example for scheduling
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

###########################################################
# 1. Load environment variables (API keys, MinIO creds, etc.)
###########################################################
load_dotenv()

def _set_env(var: str):
    """Prompt interactively if an environment variable is missing."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("FDA_API_KEY")          # If needed for your FDA API
_set_env("MINIO_ENDPOINT")       # e.g. "127.0.0.1:9000"
_set_env("MINIO_ACCESS_KEY")     # MinIO username
_set_env("MINIO_SECRET_KEY")     # MinIO password

#####################################################
# 2. Regulation Change Model & FDAMonitor
#####################################################
class RegulationChange(BaseModel):
    """
    Tracks a single regulation's metadata for change detection.
    """
    regulation_id: str
    content_hash: str
    last_updated: datetime
    details: Dict

class FDAMonitor:
    """
    Responsible for:
      - Checking MinIO for existing regulation data (if any).
      - Fetching new data from an FDA endpoint.
      - Detecting changes (via content_hash).
      - Storing new or updated records back to MinIO.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.bucket_name = "fda-regulations"  # Adjust to your environment
        
        # Instantiate Minio client
        self.minio_client = Minio(
            os.environ["MINIO_ENDPOINT"],
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            secure=False  # or True, depending on your TLS setup
        )
        
        # Ensure the target bucket exists
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

            # Convert raw JSON to RegulationChange objects
            return {k: RegulationChange(**v) for k, v in data.items()}
        except Exception:
            # If the object doesn't exist or fetch fails, return empty
            return {}

    def store_regulations(self, known_regulations: Dict[str, RegulationChange]):
        """
        Store the current set of known regulations in MinIO.
        Overwrites the 'regulations.json' object if it exists.
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
        Check FDA API for new/updated regulations.
        
        Steps:
          1. Load known regulations from MinIO (if any).
          2. Pull current (latest) data from FDA.
          3. Compare new data vs. known data (hash).
          4. If changes/new data found, store new snapshot to MinIO.
          5. Return the list of newly added or updated regulations.
        """
        # 1. Load known regs from MinIO
        known_regulations = self.fetch_stored_regulations()

        # 2. Fetch the current data from FDA (mock example endpoint)
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

        # 3. Compare data to detect changes
        changes = []
        for reg_id, content in current_regulations.items():
            current_hash = self.get_content_hash(content)
            
            # If brand new or changed, store new entry
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
                # Existing regulation - check if it changed
                if known_regulations[reg_id].content_hash != current_hash:
                    reg_change = RegulationChange(
                        regulation_id=reg_id,
                        content_hash=current_hash,
                        last_updated=datetime.now(),
                        details=content
                    )
                    known_regulations[reg_id] = reg_change
                    changes.append(reg_change)

        # 4. If any changes, re-store to MinIO
        if changes:
            self.store_regulations(known_regulations)

        # 5. Return the list of newly added/updated regs
        return changes


#######################################################
# 3. A simple function to perform the check (a "tool")
#######################################################
async def check_regulations():
    """
    Standalone function that:
      - Instantiates the FDAMonitor
      - Checks for updates
      - Prints or returns info about the changes
    """
    fda_monitor = FDAMonitor(api_key=os.environ.get("FDA_API_KEY", ""))

    changes = await fda_monitor.check_fda_updates()
    if changes:
        # Summarize the changes
        return {
            "message": f"Found {len(changes)} regulation changes.",
            "changes": [chg.dict() for chg in changes]
        }
    else:
        return {"message": "No regulation changes found."}


############################################################
# 4. Schedule the check to run periodically (e.g. once/day)
############################################################
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('interval', hours=24)  # or minutes=30, seconds=10, etc.
def scheduled_regulation_check():
    """
    APScheduler job that runs periodically.  
    Because check_regulations is async, we’ll call it via asyncio.
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(check_regulations())
    print("[Scheduled Job] ", result)

# Start the scheduler
scheduler.start()

############################################################
# 5. Example usage if run as a main script
############################################################
if __name__ == "__main__":
    # Manual check (outside of the scheduler)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(check_regulations())
    print("Manual run:", result)

    # The scheduler will keep running in the background (if desired).
    # You can also daemonize or run it in a service, etc.
