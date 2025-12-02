# API module
# REST API for iOS frontend communication:
# - Image upload endpoint
# - Pipeline status/progress
# - Result retrieval
# - WebSocket for real-time updates

from .server import create_app

__all__ = ["create_app"]
