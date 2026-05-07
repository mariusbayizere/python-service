FROM python:3.11-slim AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ \
    libglib2.0-0 libglib2.0-dev \
    libsm6 libxext6 libxrender-dev \
    libgl1-mesa-dev libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    libsdl2-dev libsdl2-mixer-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r flaskgroup && useradd -r -g flaskgroup flaskuser

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gunicorn==21.2.0 && \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-cache-dir -r requirements.txt

COPY api_server.py config.py detector.py monitor.py \
     notifications.py violation_handler.py ./

# Copy model files only if they exist
COPY store/ ./store/
COPY store_info/ ./store_info/

# Create directories for optional large files
RUN mkdir -p violations_screenshots driver_faces yolo && \
    chown -R flaskuser:flaskgroup /app

USER flaskuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api_server:app"]
