"""
FastAPI application entry point.
Run with:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, vehicle
from config import settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Intelligence Platform",
    description=(
        "Multi-modal AI system for automated vehicle service record creation.\n\n"
        "**Inputs**: CCTV vehicle image · customer text · vehicle metadata\n\n"
        "**Outputs**: vehicle type · detected damages · customer intent · service priority"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow all origins for dev; restrict in production) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request-timing middleware ──────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(vehicle.router)


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("Vehicle Intelligence Platform starting up …")
    settings.model_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Vehicle Intelligence Platform shutting down.")


# ── Root redirect ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Vehicle Intelligence Platform API", "docs": "/docs"}
