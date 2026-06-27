FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch CPU-only first (avoids CUDA bloat)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles \
    Pillow \
    opencv-python-headless \
    psutil \
    requests \
    slowapi

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p models temp output

EXPOSE 8000

ENV PORT=8000

CMD ["python", "main.py"]
