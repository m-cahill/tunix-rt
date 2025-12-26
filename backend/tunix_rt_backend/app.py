"""FastAPI application with health endpoints."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tunix_rt_backend.routers import (
    datasets,
    evaluation,
    health,
    models,
    regression,
    traces,
    tuning,
    tunix,
    tunix_runs,
    ungar,
)

# Configure logger
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tunix RT Backend",
    description="Reasoning-Trace backend with RediAI integration",
    version="0.1.0",
)

# CORS middleware for frontend integration
# M4: Allow both localhost and 127.0.0.1 for dev (5173) and preview (4173) modes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (DNS)
        "http://127.0.0.1:5173",  # Vite dev server (IPv4)
        "http://localhost:4173",  # Vite preview (DNS)
        "http://127.0.0.1:4173",  # Vite preview (IPv4)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(traces.router)
app.include_router(ungar.router)
app.include_router(datasets.router)
app.include_router(tunix.router)
app.include_router(tunix_runs.router)
app.include_router(evaluation.router)
app.include_router(regression.router)
app.include_router(tuning.router)
app.include_router(models.router)
