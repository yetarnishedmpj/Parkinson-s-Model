
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1.endpoints import router as api_router
from app.core.config import settings
import os

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router) # Support root level ws for legacy compat

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print(f"--- Protogen-Twin Backend Starting ---")
    print(f"Static Files Path: {os.path.abspath(frontend_path)}")
    print(f"Server URL: http://localhost:8000")
    print(f"API Base: {settings.API_V1_STR}")
    print(f"--------------------------------------")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
