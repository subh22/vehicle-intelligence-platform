# ── Stage 1: dependency layer ──────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System libs required by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: application layer ─────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

COPY . .

# Create runtime directories
RUN mkdir -p models data logs

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
