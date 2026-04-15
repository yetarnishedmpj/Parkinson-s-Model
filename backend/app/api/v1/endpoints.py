
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.engine import engine
from app.services.model import analyze_vitals
import asyncio
import json
from datetime import datetime

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "online", "simulation": "active"}

@router.websocket("/ws")
async def websocket_vitals(websocket: WebSocket):
    await websocket.accept()
    
    async def receive_controls():
        """Background task to swallow control messages from the client."""
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "control":
                        engine.update_controls(msg.get("x", 0), msg.get("z", 0))
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass

    # Create worker task for controls
    control_task = asyncio.create_task(receive_controls())

    try:
        while True:
            # 1. Generate Raw Vitals from Physics Engine
            reading = engine.generate_reading()
            
            # 2. Process with High-Fidelity ML Diagnostic Layer
            analytics = analyze_vitals(
                reading["heart_rate"], 
                reading["temperature"], 
                reading["activity_level"],
                reading["hazard_proximity"],
                reading["is_freezing"]
            )
            
            # 3. Combine Data and Stream to Client
            payload = {
                "timestamp": reading["timestamp"],
                "vitals": reading,
                "analytics": analytics,
                "scenario": engine.scenario
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(0.05) # 20Hz for Ultra-Smooth Motion (decreased from 100ms)
    except WebSocketDisconnect:
        print("Telemetry client disconnected")
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        control_task.cancel()
        try:
            await websocket.close()
        except:
            pass

@router.post("/scenario/{name}")
async def set_scenario(name: str):
    engine.set_scenario(name)
    return {"status": "success", "new_scenario": engine.scenario}

# Legacy endpoint - retained for compatibility but deprecated
@router.post("/control/move")
async def manual_move(req: dict):
    engine.update_controls(req.get("x", 0), req.get("z", 0))
    return {"status": "success", "deprecated": True}
