import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional
import httpx  # Use httpx for async POST forwarding

# If your FastAPI server is running on port 8675, you can do:
# curl -X POST "http://127.0.0.1:8675/api/upload/documents" \
#   -F file=@/Users/guybeals/Downloads/paper04_textual_resource.pdf \
#   -F subfolder="pdf"

router = APIRouter()

# points the request to be forwarded to the other container
DOC_PIPELINE_URL = "http://172.17.0.5:3009/docs/upload/documents"

@router.post("/documents")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subfolder: Optional[str] = None
):
    """
    Pass-through endpoint:
      1) Receives an uploaded file.
      2) Immediately forwards it to the doc pipeline at DOC_PIPELINE_URL.
      3) Returns the pipeline's response as JSON.
    """
    original_filename = file.filename
    if not original_filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # We'll read the file contents into memory (caution for very large files)
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    # Build the multipart form data for forwarding
    files = {
        "file": (original_filename, contents, file.content_type),
    }
    data = {}
    if subfolder:
        data["subfolder"] = subfolder

    # Forward the request to the doc pipeline
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                DOC_PIPELINE_URL,
                files=files,
                data=data,
                timeout=60.0  # adjust as needed
            )
            response.raise_for_status()  # Raises HTTPError if status 4xx/5xx
        except httpx.RequestError as e:
            # Network error, DNS failure, etc.
            raise HTTPException(status_code=500, detail=f"Error connecting to doc pipeline: {e}")
        except httpx.HTTPStatusError as e:
            # Non-200 HTTP response
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Doc pipeline responded with error: {e.response.text}"
            )

    # (Optional) do any background tasks here
    background_tasks.add_task(log_forward_event, original_filename)

    # Return the doc pipeline's JSON response directly
    # If your doc pipeline doesn't return JSON, adjust accordingly
    return {
        "message": "File forwarded successfully to document pipeline",
        "doc_pipeline_response": response.json()
    }

async def log_forward_event(filename: str):
    """
    Simple example of a background task after forwarding the file.
    """
    await asyncio.sleep(0.1)
    print(f"[log_forward_event] Successfully forwarded file: {filename}")

@router.get("/health")
async def upload_info():
    return {"message": "Upload pass-through is ready."}