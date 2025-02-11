# initializers.py

import logging
import os
from src.model.query_model import QueryModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global cache for the embedding model.

QUERY_MODEL_CACHE = {}

def warm_up_query_model(model_name: str, device: str = None) -> QueryModel:
    """
    Create and warm up the QueryModel, returning the ready instance.
    If model_name is None, we rely on the config to provide it.
    """
    global QUERY_MODEL_CACHE
    if model_name in QUERY_MODEL_CACHE:
        logger.info(f"Using cached query model for: {model_name}")
        return QUERY_MODEL_CACHE[model_name]

    logger.info(f"Warming up query model: {model_name}...") 
    query_model = QueryModel(model_name=model_name, device=device)
    logger.info("Initializing the QueryModel...")
    
    QUERY_MODEL_CACHE[model_name] = query_model
    query_model.warm_up()
    return query_model

# if __name__ == "__main__":
#     # For testing purposes, you can run this file directly.
#     # Replace 'default_model' with the model name specified in your config if needed.
#     model_name = "default_model"
#     warm_up_embedder(model_name)
