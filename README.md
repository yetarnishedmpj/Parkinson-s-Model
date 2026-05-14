# Protogen-Twin 🧬🤖

A high-fidelity Digital Twin platform designed for real-time vitals monitoring and simulation. Protogen-Twin provides a seamless integration between physics-based engines (supporting Parkinson's vitals simulation) and a modern telemetry dashboard.

## 🚀 Overview

Protogen-Twin allows for the real-time visualization of high-frequency telemetry data. It features dual backend implementations (Python and C#) to demonstrate modularity and performance across different tech stacks, both serving a shared vanilla JavaScript frontend.

## 🛠️ Tech Stack

### Python Backend
- **FastAPI**: Modern, fast web framework for building APIs.
- **WebSockets**: Real-time bidirectional communication for telemetry.
- **Pydantic**: Data validation and settings management.
- **Uvicorn**: Lightning-fast ASGI server implementation.

### C# Backend
- **ASP.NET Core**: High-performance, open-source web framework.
- **Minimal APIs**: Streamlined approach to building HTTP services.
- **System.Net.WebSockets**: Native WebSocket support for telemetry streaming.

### Frontend
- **HTML5 / CSS3**: Modern responsive styling with a focus on telemetry visualization.
- **Vanilla JavaScript**: Lightweight, high-performance logic for real-time data rendering.

## 📂 Project Structure

```text
drive-download/
├── backend/            # Python FastAPI Implementation
│   ├── app/            # Core logic, API endpoints, and services
│   └── main.py         # Python application entry point
├── backend_cs/         # C# .NET Implementation
│   ├── Models/         # Data structures
│   ├── Services/       # Parkinson's Physics Engine logic
│   └── Program.cs      # C# application entry point
└── frontend/           # Shared Vanilla JS Dashboard
    ├── index.html      # Main dashboard interface
    ├── app.js          # Telemetry visualization logic
    └── styles.css      # Dashboard styling
```

## ⚙️ Getting Started

### 1. Python Backend Setup
```bash
cd backend
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/macOS
# Install dependencies
pip install fastapi uvicorn pydantic-settings
# Run the server
python main.py
```
The server will be available at `http://localhost:8000`.

### 2. C# Backend Setup
```bash
cd backend_cs
# Restore and run the project
dotnet run
```
The server will be available at `http://localhost:5000` (or as configured in your launch settings).

### 3. Accessing the Dashboard
Once either backend is running, open your browser and navigate to the server URL (e.g., `http://localhost:8000`). The backend automatically serves the frontend static files.

## 📡 API Endpoints

- `GET /api/v1/status`: Check system health.
- `WS /ws`: Real-time telemetry stream (Core functionality).
- `POST /api/v1/scenario/{name}`: Change the active simulation scenario (e.g., `extreme`, `stable`).
- `POST /api/v1/control/move`: Manual movement control (deprecated in favor of WebSocket control).

---
*Protogen-Twin - Bridging Reality and Simulation.*
