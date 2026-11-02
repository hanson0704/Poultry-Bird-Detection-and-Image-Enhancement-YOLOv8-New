"""FastAPI application entrypoint for Poultry Detection & Behaviour Analysis."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_image, routes_model, routes_video
from app.state import get_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_detector()
    except Exception:
        # Health/model endpoints report the exact error if best.pt is missing or invalid
        pass
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Based Poultry Detection and Behaviour Analysis API",
        version="1.0.0",
        description=(
            "Integrates YOLOv8m chicken detection (best.pt) with CentroidTracker, "
            "KDTree spatial density, Euclidean movement (TDP), and sliding-window "
            "Inactivity / Low-Mobility Indicator analysis."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_model.router)
    app.include_router(routes_image.router)
    app.include_router(routes_video.router)
    return app


app = create_app()

