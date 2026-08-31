FROM python:3.12-slim

# opencv-*-headless still links libglib2.0 even though it drops the GUI/GL stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use the fully pinned lock file for a reproducible image.
COPY requirements.txt requirements-lock.txt requirements-ai.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY app ./app
COPY scripts ./scripts
RUN mkdir -p uploads processed cache

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
