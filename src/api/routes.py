from fastapi import FastAPI
from api.endpoints.knowledge_base_endpoints import router


def setup_routes(app: FastAPI):
    app.include_router(router, tags=["knowledge_base"])