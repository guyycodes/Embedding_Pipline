# api/upload_controller.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/upload")
async def upload():
    """
    Simple endpoint returning "Hello world".
    """
    return {"message": "Hello world"}