"""Entrypoint for running the NFT Recommender micro‑service.

Usage
-----
uvicorn main:app --host 0.0.0.0 --port 8000

FastAPI automatically exposes Swagger UI at /docs and ReDoc at /redoc.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from App.routes import router

app = FastAPI(
    title="NFT Recommender Service",
    version="1.2.0",
    description="API for NFT recommendation and content-based search with user authentication",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc",  # ReDoc docs
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router with a prefix
app.include_router(router, prefix="/api")
