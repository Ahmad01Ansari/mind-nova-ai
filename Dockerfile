FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg which is REQUIRED for Faster-Whisper to process audio
# Also install build-essential, pkg-config and libav in case PyAV needs to compile from source
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential pkg-config libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Run the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
