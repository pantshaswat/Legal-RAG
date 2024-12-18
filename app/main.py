from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from models.models_init import init_models
from app.services.qdrant_client_init import get_qdrant_client
def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal RAG Search API",
        description="RAG-powered legal document search",
        version="0.1.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(api_router, prefix="/api/v1")

    return app


client = get_qdrant_client()


app = create_app()