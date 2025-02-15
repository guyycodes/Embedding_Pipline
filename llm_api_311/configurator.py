#!/usr/bin/env python3
"""
configurator.py

A CLI script for:
1) Updating the config.yml with user-selected settings (only those found in your config.yml).
2) Interacting with the QueryModel for QA or deep semantic searches.
"""

import yaml
import questionary
import logging
from pathlib import Path
import sys
import os

# Ensure the project root is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.query_model import QueryModel  # Assuming you still have QueryModel
from src.util.get_agent_config import load_agent_config
from src.vector_store.qdrant_config import QdrantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yml"


def load_config(config_path: str = CONFIG_FILENAME) -> dict:
    """
    Loads YAML config from disk. Returns an empty dict if file not found.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_file} not found. Starting with empty config.")
        return {}
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config_data: dict, config_path: str = CONFIG_FILENAME):
    """
    Saves config data to the specified YAML file.
    """
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False)
    logger.info(f"✅ Configuration saved to {config_path}.")


def update_configuration():
    """
    Prompts the user for the config keys that exist in the current config.yml and updates them.
    """
    print("\n=== Update Configuration ===\n")

    # Load existing config
    current_config = load_config(CONFIG_FILENAME)

    # Ensure required top-level keys exist
    if "agent" not in current_config:
        current_config["agent"] = {}
    if "query_model" not in current_config["agent"]:
        current_config["agent"]["query_model"] = {}
    if "qdrant" not in current_config:
        current_config["qdrant"] = {}

    # --- AGENT MODEL NAME ---
    model_name = questionary.text(
        "Enter the agent model name (e.g., guymorganb/e5-large-v2-4096-lsg-patched):",
        default=str(current_config["agent"].get("model_name", ""))
    ).ask()
    current_config["agent"]["model_name"] = model_name

    # --- QUERY MODEL SETTINGS ---
    query_model_cfg = current_config["agent"].get("query_model", {})
    device = questionary.select(
        "Select device for the query model:",
        choices=["cpu", "cuda", "mps"],
        default=str(query_model_cfg.get("device", "cpu"))
    ).ask()
    current_config["agent"]["query_model"]["device"] = device

    max_tokens = questionary.text(
        "Enter max_tokens for the query model:",
        default=str(query_model_cfg.get("max_tokens", "4096"))
    ).ask()
    current_config["agent"]["query_model"]["max_tokens"] = int(max_tokens)

    timeout = questionary.text(
        "Enter timeout (in seconds) for the query model:",
        default=str(query_model_cfg.get("timeout", "60"))
    ).ask()
    current_config["agent"]["query_model"]["timeout"] = int(timeout)

    top_k = questionary.text(
        "Enter top_k for query searches:",
        default=str(query_model_cfg.get("top_k", "5"))
    ).ask()
    current_config["agent"]["query_model"]["top_k"] = int(top_k)

    # --- QDRANT SETTINGS ---
    qdrant_cfg = current_config.get("qdrant", {})

    collection = questionary.text(
        "Enter Qdrant collection name:",
        default=str(qdrant_cfg.get("collection", "document_vectors"))
    ).ask()
    current_config["qdrant"]["collection"] = collection

    cloud_host = questionary.text(
        "Enter Qdrant cloud_host (if applicable, otherwise leave blank):",
        default=str(qdrant_cfg.get("cloud_host", ""))
    ).ask()
    current_config["qdrant"]["cloud_host"] = cloud_host

    cloud_port = questionary.text(
        "Enter Qdrant cloud_port (443 by default for TLS):",
        default=str(qdrant_cfg.get("cloud_port", "443"))
    ).ask()
    current_config["qdrant"]["cloud_port"] = int(cloud_port)

    cloud_api_key = questionary.text(
        "Enter Qdrant cloud_api_key (leave blank if not using cloud):",
        default=str(qdrant_cfg.get("cloud_api_key", ""))
    ).ask()
    current_config["qdrant"]["cloud_api_key"] = cloud_api_key

    prefer_grpc = questionary.select(
        "Prefer gRPC for Qdrant connection?",
        choices=["true", "false"],
        default="true" if qdrant_cfg.get("prefer_grpc", False) else "false"
    ).ask()
    current_config["qdrant"]["prefer_grpc"] = (prefer_grpc.lower() == "true")

    dimension = questionary.text(
        "Enter vector dimension for Qdrant:",
        default=str(qdrant_cfg.get("dimension", "1024"))
    ).ask()
    current_config["qdrant"]["dimension"] = int(dimension)

    host = questionary.text(
        "Enter Qdrant host:",
        default=str(qdrant_cfg.get("host", "localhost"))
    ).ask()
    current_config["qdrant"]["host"] = host

    mode = questionary.select(
        "Select Qdrant mode:",
        choices=["LOCAL", "CLOUD"],
        default=str(qdrant_cfg.get("mode", "LOCAL"))
    ).ask()
    current_config["qdrant"]["mode"] = mode

    port = questionary.text(
        "Enter Qdrant port:",
        default=str(qdrant_cfg.get("port", "6333"))
    ).ask()
    current_config["qdrant"]["port"] = int(port)

    qdrant_timeout = questionary.text(
        "Enter Qdrant timeout (in seconds):",
        default=str(qdrant_cfg.get("timeout", "300"))
    ).ask()
    current_config["qdrant"]["timeout"] = int(qdrant_timeout)

    # Save the updated config
    save_config(current_config, CONFIG_FILENAME)
    print("Configuration updated successfully.\n")


def test_query_model():
    """
    Demonstrates an interactive QueryModel usage:
      1) Loads the config to find the Qdrant collection name.
      2) Instantiates + warms up QueryModel.
      3) Asks user if they want a QA search or deep semantic search.
      4) Prompts for query, runs the chosen search, displays results.
    """
    print("\n=== Test Query Model ===\n")

    # 1) Load config and figure out Qdrant collection name (default to "document_vectors" if missing).
    config = load_config(CONFIG_FILENAME)
    qdrant_cfg = config.get("qdrant", {})
    collection_name = qdrant_cfg.get("collection", "document_vectors")

    # 2) Instantiate + warm up the QueryModel
    try:
        query_model = QueryModel()
        query_model.warm_up()
    except Exception as e:
        print(f"❌ Error initializing QueryModel: {e}")
        return

    while True:
        # 3) Ask user which search type
        search_type = questionary.select(
            "Select a query type:",
            choices=[
                "QA Search",
                "Deep Semantic Search",
                "Exit to main menu"
            ]
        ).ask()

        if search_type == "Exit to main menu":
            print("Returning to main menu...\n")
            break

        # 4) Prompt for the user query
        user_query = questionary.text(
            "Enter your query (type 'quit' to exit):"
        ).ask()

        if user_query.lower() in ("quit", "exit"):
            print("Returning to main menu...\n")
            break

        # 5) Perform the chosen search
        if search_type == "QA Search":
            results = query_model.qa_search(user_query, collection_name)
        else:  # "Deep Semantic Search"
            results = query_model.deep_semantic_search(user_query, collection_name)

        # 6) Display results
        print("\n=== Search Results ===")
        if not results:
            print("No results returned.")
        else:
            for idx, r in enumerate(results, 1):
                print(f"[{idx}] {r}")
        print("\n")


def main_menu():
    """
    The main CLI menu.
    """
    while True:
        choice = questionary.select(
            "Select an action:",
            choices=[
                "Update Configuration",
                "Test Query Model",
                "Exit"
            ]
        ).ask()

        if choice == "Update Configuration":
            update_configuration()
        elif choice == "Test Query Model":
            test_query_model()
        else:
            print("Exiting CLI. Goodbye!")
            break


if __name__ == "__main__":
    main_menu()
