# controllers/config_controller.py

from fastapi import APIRouter, HTTPException
import yaml
import os

router = APIRouter()

# Adjust to your actual config location or store it in this separate project
CONFIG_PATH = os.path.join("src", "config.yml")

@router.post("/updateConfig")
async def update_config(updates: dict):
    """
    Updates the config.yml with the provided values.
    {
        "agent": {
            "embedding_model": {
                "batch_size": 64
            },
        },
        "qdrant": {
            "port": 7000
        }
    }
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    errors = []

    def recursive_update(current: dict, changes: dict, path: str = ""):
        for key, value in changes.items():
            current_path = f"{path}.{key}" if path else key
            if key not in current:
                errors.append(f"Key '{current_path}' does not exist.")
            else:
                if isinstance(current[key], dict) and isinstance(value, dict):
                    recursive_update(current[key], value, current_path)
                else:
                    expected_type = type(current[key])
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"Wrong type for key '{current_path}'. "
                            f"Expected {expected_type.__name__}, got {type(value).__name__}."
                        )
                    else:
                        current[key] = value

    recursive_update(config, updates)

    if errors:
        raise HTTPException(status_code=400, detail=errors)

    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")

    return {"message": "Config updated successfully", "config": config}
