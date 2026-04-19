from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.engine import engine
from app.services.model  import analyze_vitals
from app.services.session_store import get_store
import asyncio, json
from datetime import datetime
from uuid import uuid4

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "online", "simulation": "active"}

@router.websocket("/ws")
async def websocket_vitals(websocket: WebSocket):
    await websocket.accept()
    store      = get_store()
    session_id = store.new_session()
    tick_count = 0

    async def receive_controls():
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

    control_task = asyncio.create_task(receive_controls())

    try:
        while True:
            reading   = engine.generate_reading()
            analytics = analyze_vitals(
                reading["heart_rate"],
                reading["temperature"],
                reading["activity_level"],
                reading["hazard_proximity"],
                reading["is_freezing"],
                reading.get("fatigue", 0.0),
            )

            payload = {
                "timestamp": reading["timestamp"],
                "vitals":    reading,
                "analytics": analytics,
                "scenario":  engine.scenario,
                "session_id": session_id,
            }

            await websocket.send_json(payload)

            # Persist tick (every tick, but skip freeze_heatmap to keep DB lean)
            tick_vitals = {k: v for k, v in reading.items()
                           if k not in ("npcs", "freeze_heatmap")}
            store.log_tick(session_id, reading["timestamp"], tick_vitals, analytics)
            tick_count += 1

            await asyncio.sleep(0.05)   # 20 Hz

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected after {tick_count} ticks.")
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        store.close_session(session_id)
        control_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/scenario/{name}")
async def set_scenario(name: str):
    engine.set_scenario(name)
    return {"status": "success", "new_scenario": engine.scenario}


@router.get("/mall")
async def get_mall():
    return {"obstacles": engine.obstacles, "escalators": engine.escalators}


# ── Session replay endpoints ────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions():
    return get_store().list_sessions()


@router.get("/sessions/{session_id}/replay")
async def get_replay(session_id: str):
    ticks = get_store().get_session(session_id)
    return {"session_id": session_id, "tick_count": len(ticks), "ticks": ticks}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    get_store().delete_session(session_id)
    return {"status": "deleted"}


# Legacy — retained for compatibility
@router.post("/control/move")
async def manual_move(req: dict):
    engine.update_controls(req.get("x", 0), req.get("z", 0))
    return {"status": "success", "deprecated": True}
