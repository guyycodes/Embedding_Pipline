# controllers/__init__.py

from fastapi import APIRouter
from .config_controller import router as config_router
from .upload_controller import router as upload_router
from .query_controller import router as query_router

api_router = APIRouter()
api_router.include_router(config_router, prefix="/config", tags=["config"])
api_router.include_router(upload_router, prefix="/upload", tags=["upload"])
api_router.include_router(query_router, prefix="/models", tags=["query"])