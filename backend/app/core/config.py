
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Digital Twin MVP"
    API_V1_STR: str = "/api/v1"
    WS_STR: str = "/ws"
    
    # Simulation settings
    SIMULATION_INTERVAL_SEC: float = 1.0
    
    class Config:
        case_sensitive = True

settings = Settings()
