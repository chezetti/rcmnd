"""Entrypoint for running the NFT Recommender micro‑service.

Usage
-----
uvicorn main:app --host 0.0.0.0 --port 8000

FastAPI automatically exposes Swagger UI at /docs and ReDoc at /redoc.
"""
from __future__ import annotations

from fastapi import FastAPI

from App.routes import router

app = FastAPI(
    title="NFT Recommender Service",
    version="1.1.0",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc",  # ReDoc docs
)

app.include_router(router)
