# app/services/qdrant_client_init.py
from qdrant_client import QdrantClient

def get_qdrant_client() -> QdrantClient:
    """
    Initialize and return a Qdrant client instance.
    """
    return QdrantClient(
        host="localhost",  # Docker default host
        port=6333,         # Default Qdrant port
        prefer_grpc=True   # Recommended for better performance
    )
