
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class VitalsReading(BaseModel):
    heart_rate: float = Field(..., example=72.5)
    temperature: float = Field(..., example=36.6)
    activity_level: float = Field(..., example=0.1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SimulationState(BaseModel):
    is_active: bool
    scenario: str
    current_reading: Optional[VitalsReading] = None

class HealthDiagnostics(BaseModel):
    stress_level: float
    health_index: float
    status: str
