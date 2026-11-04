import os
import modal
from dotenv import load_dotenv

load_dotenv()

# Define the Modal application
app = modal.App("ai-sprint-coach")

# Define the container image with all system & Python dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "curl")
    .pip_install(
        "fastapi",
        "uvicorn",
        "python-multipart",
        "jinja2",
        "mediapipe",
        "numpy",
        "opencv-python",
        "transformers",
        "gdown",
        "matplotlib",
        "pandas",
        "scipy",
        "groq",
        "python-dotenv",
        "timm",
        "einops",
    )
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cpu"
    )
    # Pre-download models in image layer so startup is instant
    .run_commands(
        "mkdir -p /root/models /root/data /root/uploads",
        "curl -L -o /root/models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "python -c 'import gdown; gdown.download(id=\"1H4tq_oHQkGbYqZxS_5SgOqNicRYA1zsI\", output=\"/root/models/TCPFormer_ap3d_81.pth.tr\", quiet=False)'"
    )
    # Sync application code and assets
    .add_local_dir("static", remote_path="/root/static")
    .add_local_dir("templates", remote_path="/root/templates")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("TCPFormer", remote_path="/root/TCPFormer")
    .add_local_file("pipeline.py", remote_path="/root/pipeline.py")
    .add_local_file("app.py", remote_path="/root/app.py")
)

# Use GROQ_API_KEY from local .env, or fallback to Modal Dashboard secret
groq_key = os.environ.get("GROQ_API_KEY", "")
secrets = []
if groq_key:
    secrets.append(modal.Secret.from_dict({"GROQ_API_KEY": groq_key}))
else:
    secrets.append(modal.Secret.from_name("groq-secret"))

# Deploy the FastAPI application as a 24/7 web endpoint
@app.function(
    image=image,
    secrets=secrets,
    memory=4096,  # 4 GB RAM for PyTorch & TCPFormer
    cpu=2.0,      # 2 vCPUs
    timeout=600,  # 10 min request timeout
)
@modal.asgi_app()
def fastapi_app():
    import os
    os.chdir("/root")
    from app import app as web_app
    return web_app
