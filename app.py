import os
import uuid
import math
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load GROQ_API_KEY and other env vars from .env
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import numpy as np

from pipeline import analyze_video_pipeline

app = FastAPI(title="AI Sports Technique Coach")

# Directories
os.makedirs("static",  exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Thread pool so the heavy ML pipeline doesn't block the event loop
executor = ThreadPoolExecutor(max_workers=1)


def sanitize_for_json(obj):
    """
    Recursively converts NaN, Infinity, and numpy types to JSON-compliant values.
    Prevents ValueError: Out of range float values are not JSON compliant: nan.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return 0.0 if math.isnan(val) or math.isinf(val) else val
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """Health check for server and API keys."""
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "healthy",
        "groq_configured": has_groq,
        "models": {
            "pose_landmarker": os.path.exists(os.path.join("models", "pose_landmarker_full.task")),
            "tcpformer": os.path.exists(os.path.join("models", "TCPFormer_ap3d_81.pth.tr")),
            "reference_data": os.path.exists(os.path.join("data", "Running_0_cam_1_TCPFormer_features_normalized.csv"))
        }
    }


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts an uploaded video, runs the full ML pipeline,
    and returns structured JSON results.
    """
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_location = os.path.join("uploads", unique_name)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the heavy CPU/GPU pipeline in a thread pool
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            executor,
            analyze_video_pipeline,
            file_location
        )

        # Clean all NaNs and numpy types so Starlette JSONResponse never fails
        clean_results = sanitize_for_json(results)
        return JSONResponse(content=clean_results)

    except Exception as e:
        print("[ERROR] Exception in /analyze endpoint:")
        traceback.print_exc()
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )
    finally:
        # Clean up uploaded file after processing
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

