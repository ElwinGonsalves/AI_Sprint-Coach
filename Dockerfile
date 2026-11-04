FROM python:3.12-slim

# Install system dependencies for OpenCV and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies as user
COPY --chown=user:user requirements.txt .

# Install CPU-only PyTorch first (much smaller & faster build), then remaining packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=user:user . .

# Pre-download ML checkpoints during build so app starts immediately
RUN python setup_ml.py

# Hugging Face Spaces port
EXPOSE 7860

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

