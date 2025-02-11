# controllers/query_controller.py

import logging
import functools
import asyncio
import threading
import uuid
import time
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel


from src.models.query_model import QueryModel
from src.vector_store.qdrant_config import QdrantManager
from src.util.get_agent_config import load_agent_config
from initializers import warm_up_query_model

logger = logging.getLogger(__name__)
router = APIRouter()

class QdrantConnectionPool:
    """Example Qdrant connection pool."""
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.pool: Dict[str, list] = {}
        self.pool_locks: Dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def get_connection(self, collection_name: str):
        thread_name = threading.current_thread().name
        logger.info(f"[QdrantPool] Trying to get connection for '{collection_name}' on thread={thread_name}")

        if collection_name not in self.pool:
            self.pool[collection_name] = []
        if collection_name not in self.pool_locks:
            self.pool_locks[collection_name] = asyncio.Lock()

        async with self.pool_locks[collection_name]:
            if not self.pool[collection_name]:
                if len(self.pool[collection_name]) < self.max_size:
                    conn = QdrantManager(collection_name)
                    self.pool[collection_name].append(conn)
                    logger.info(f"[QdrantPool] Created new QdrantManager for '{collection_name}'.")
                else:
                    logger.error("[QdrantPool] Connection pool exhausted!")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Connection pool for '{collection_name}' is exhausted."
                    )
            connection = self.pool[collection_name].pop()
            logger.info(f"[QdrantPool] Got connection from pool for '{collection_name}'.")

        try:
            yield connection
        finally:
            async with self.pool_locks[collection_name]:
                self.pool[collection_name].append(connection)
                logger.info(f"[QdrantPool] Returned connection to pool for '{collection_name}'.")

###############################################################################
# GLOBAL STATE
###############################################################################
class GlobalState:
    def __init__(self, max_workers: int = 10, pool_size: int = 20):
        # Cache of model_name -> QueryModel
        self.query_models: Dict[str, QueryModel] = {}
        # Lock for each model_name to prevent concurrent initializations
        self.model_locks: Dict[str, asyncio.Lock] = {}

        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Qdrant connection pool
        self.qdrant_pool = QdrantConnectionPool(max_size=pool_size)

global_state = GlobalState()

###############################################################################
# REQUEST MODELS
###############################################################################
class QARequest(BaseModel):
    query: str
    collection_name: str = "document_vectors"

###############################################################################
# HELPER FUNCTIONS
###############################################################################
async def async_load_config(request_id: str):
    """
    Load the agent config in a thread pool so it doesn't block the event loop.
    Logs the request_id to help track concurrency.
    """
    thread_name = threading.current_thread().name
    logger.info(f"[{request_id}] [async_load_config] START on thread={thread_name}")
    loop = asyncio.get_event_loop()
    cfg = await loop.run_in_executor(global_state.thread_pool, load_agent_config)
    logger.info(f"[{request_id}] [async_load_config] DONE on thread={thread_name}")
    return cfg

async def get_or_create_query_model(model_name: str, device: str, request_id: str) -> QueryModel:
    """
    Asynchronously fetch or create a QueryModel, ensuring only one coroutine
    initializes the model at a time. Logs for concurrency tracing.
    """
    thread_name = threading.current_thread().name
    logger.info(f"[{request_id}] [get_or_create_query_model] Checking model cache for '{model_name}' (device={device}) on thread={thread_name}")

    if model_name not in global_state.model_locks:
        global_state.model_locks[model_name] = asyncio.Lock()

    async with global_state.model_locks[model_name]:
        if model_name not in global_state.query_models:
            logger.info(f"[{request_id}] [get_or_create_query_model] Model '{model_name}' NOT in cache. Initializing now...")
            loop = asyncio.get_event_loop()
            query_model = await loop.run_in_executor(
                global_state.thread_pool,
                functools.partial(warm_up_query_model, model_name=model_name, device=device)
            )
            global_state.query_models[model_name] = query_model
            logger.info(f"[{request_id}] [get_or_create_query_model] Model '{model_name}' loaded and cached.")
        else:
            logger.info(f"[{request_id}] [get_or_create_query_model] Model '{model_name}' is already in cache.")

    return global_state.query_models[model_name]

async def check_qdrant_collection(qdrant: QdrantManager, request_id: str) -> bool:
    """
    Check if a Qdrant collection exists. Offloaded to the thread pool.
    Logs for concurrency tracing.
    """
    thread_name = threading.current_thread().name
    logger.info(f"[{request_id}] [check_qdrant_collection] START on thread={thread_name}")
    loop = asyncio.get_event_loop()
    exists = await loop.run_in_executor(
        global_state.thread_pool,
        qdrant.collection_exists
    )
    logger.info(f"[{request_id}] [check_qdrant_collection] Collection exists={exists} on thread={thread_name}")
    return exists

###############################################################################
# QA ENDPOINT
###############################################################################
@router.post("/qa")
async def run_qa_search(request: QARequest, background_tasks: BackgroundTasks):
    """
    Asynchronous QA endpoint with concurrency support.
    """
    # Generate a unique request ID to trace the logs
    request_id = f"QA-{uuid.uuid4().hex[:6]}"
    start_time = time.time()

    thread_name = threading.current_thread().name
    logger.info(f"========== START /qa request_id={request_id}, thread={thread_name}, query='{request.query}' ==========")
    logger.info(f"[{request_id}] Active tasks in event loop: {len(asyncio.all_tasks())}")

    try:
        # 1. Load config asynchronously
        config = await async_load_config(request_id)
        agent_section = config.get("agent", {})
        model_name = agent_section.get("model_name", "guymorganb/e5-large-v2-4096-lsg-patched")
        device = agent_section.get("embedding_model", {}).get("device", "cpu")

        # 2. Retrieve or create the model
        query_model = await get_or_create_query_model(model_name, device, request_id)

        # 3. Acquire a Qdrant connection from our pool
        async with global_state.qdrant_pool.get_connection(request.collection_name) as qdrant:
            # 3a. Check if the collection exists
            exists = await check_qdrant_collection(qdrant, request_id)
            if not exists:
                logger.error(f"[{request_id}] Qdrant collection '{request.collection_name}' does NOT exist.")
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{request.collection_name}' not found in Qdrant."
                )

            # 4. Run the QA search in the thread pool
            search_thread = threading.current_thread().name
            logger.info(f"[{request_id}] [run_qa_search] Starting 'qa_search' on thread={search_thread}")
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                global_state.thread_pool,
                functools.partial(
                    query_model.qa_search,
                    question=request.query,
                    collection_name=request.collection_name
                )
            )
            logger.info(f"[{request_id}] [run_qa_search] Finished 'qa_search' on thread={search_thread}. Found {len(docs)} docs.")

            # 5. Log a completion message as a background task (doesn't block the response)
            background_tasks.add_task(
                logger.info, 
                f"[{request_id}] [QA Search BackgroundTask] Query='{request.query}' completed with {len(docs)} docs."
            )

            elapsed = time.time() - start_time
            logger.info(f"========== END /qa request_id={request_id}, elapsed={elapsed:.2f}s ==========")

            # 6. Return the documents
            return {"docs": docs, "request_id": request_id, "elapsed_seconds": round(elapsed, 2)}

    except Exception as e:
        logger.error(f"[{request_id}] Error during QA search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"[{request_id}] Error during QA search: {str(e)}"
        )

###############################################################################
# SEMANTIC ENDPOINT
###############################################################################
@router.post("/semantic")
async def run_semantic_search(request: QARequest, background_tasks: BackgroundTasks):
    """
    Asynchronous deep semantic search endpoint with concurrency support.
    """
    # Generate a unique request ID to trace the logs
    request_id = f"SEM-{uuid.uuid4().hex[:6]}"
    start_time = time.time()

    thread_name = threading.current_thread().name
    logger.info(f"========== START /semantic request_id={request_id}, thread={thread_name}, query='{request.query}' ==========")
    logger.info(f"[{request_id}] Active tasks in event loop: {len(asyncio.all_tasks())}")

    try:
        # 1. Load config asynchronously
        config = await async_load_config(request_id)
        agent_section = config.get("agent", {})
        model_name = agent_section.get("model_name", "guymorganb/e5-large-v2-4096-lsg-patched")
        device = agent_section.get("embedding_model", {}).get("device", "cpu")

        # 2. Retrieve or create the model
        query_model = await get_or_create_query_model(model_name, device, request_id)

        # 3. Acquire a Qdrant connection from our pool
        async with global_state.qdrant_pool.get_connection(request.collection_name) as qdrant:
            # 3a. Check if the collection exists
            exists = await check_qdrant_collection(qdrant, request_id)
            if not exists:
                logger.error(f"[{request_id}] Qdrant collection '{request.collection_name}' does NOT exist.")
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{request.collection_name}' not found in Qdrant."
                )

            # 4. Run the deep semantic search in the thread pool
            search_thread = threading.current_thread().name
            logger.info(f"[{request_id}] [run_semantic_search] Starting 'deep_semantic_search' on thread={search_thread}")
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                global_state.thread_pool,
                functools.partial(
                    query_model.deep_semantic_search,
                    query=request.query,
                    collection_name=request.collection_name
                )
            )
            logger.info(f"[{request_id}] [run_semantic_search] Finished 'deep_semantic_search' on thread={search_thread}. Found {len(docs)} docs.")

            # 5. Log or do cleanup in a background task
            background_tasks.add_task(
                logger.info,
                f"[{request_id}] [Semantic Search BackgroundTask] Query='{request.query}' completed with {len(docs)} docs."
            )

            elapsed = time.time() - start_time
            logger.info(f"========== END /semantic request_id={request_id}, elapsed={elapsed:.2f}s ==========")

            # 6. Return the documents
            return {"docs": docs, "request_id": request_id, "elapsed_seconds": round(elapsed, 2)}

    except Exception as e:
        logger.error(f"[{request_id}] Error during semantic search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"[{request_id}] Error during semantic search: {str(e)}"
        )
