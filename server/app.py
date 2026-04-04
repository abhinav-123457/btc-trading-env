"""
app.py — FastAPI application wrapping the BTC Trading Environment.

Exposes both HTTP REST endpoints (required by the competition pre-flight check)
and a WebSocket endpoint (used by the OpenEnv EnvClient).

HTTP endpoints:
  POST /reset           body: {task_id?, seed?}   → TradingObservation
  POST /step            body: TradeAction           → StepResult
  GET  /state                                       → TradingState
  GET  /health                                      → {"status": "ok"}
  GET  /tasks                                       → list of available tasks
  POST /grade           body: {}                    → GraderResult

WebSocket:
  WS /ws                streaming session-based interface
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import uuid
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import TASK_CONFIGS, TradingEnvironment
from models import TradeAction, TradingObservation, TradingState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("btc-trading-env")


# App & session store


app = FastAPI(
    title="BTC Algorithmic Trading OpenEnv",
    description=(
        "A Bitcoin trading execution environment for RL agent training. "
        "Implements the OpenEnv spec: step(), reset(), state()."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment for HTTP endpoints (stateful per-session)
# In production, use session IDs → per-session environments
_http_env = TradingEnvironment()
_http_initialized = False


# Request / response schemas


class ResetRequest(BaseModel):
    task_id: str = "easy_profitable_baseline"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str = "hold"
    amount: float = 0.0
    limit_price: Optional[float] = None



# HTTP endpoints


@app.get("/health")
async def health_check():
    """Health probe — must return HTTP 200."""
    return {"status": "ok", "environment": "btc-trading-env", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    """Return metadata for all available tasks."""
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": ["easy", "medium", "hard"][i],
                **{k: v for k, v in cfg.items()},
            }
            for i, (tid, cfg) in enumerate(TASK_CONFIGS.items())
        ]
    }


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> JSONResponse:
    """
    Initialise a new episode.
    Returns the first TradingObservation as JSON.
    """
    global _http_initialized
    task_id = request.task_id if request else "easy_profitable_baseline"
    seed = request.seed if request else None
    obs = _http_env.reset(task_id=task_id, seed=seed)
    _http_initialized = True
    return JSONResponse(content=obs.model_dump(), status_code=200)


@app.post("/step")
async def step(request: StepRequest) -> JSONResponse:
    """Execute one action and return StepResult."""
    global _http_initialized
    if not _http_initialized:
        _http_env.reset()
        _http_initialized = True

    action = TradeAction(
        action_type=request.action_type,
        amount=request.amount,
        limit_price=request.limit_price,
    )
    obs, reward, done, info = _http_env.step(action)
    return JSONResponse(
        content={
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        },
        status_code=200,
    )


@app.get("/state")
async def state() -> JSONResponse:
    """Return current episode state metadata."""
    global _http_initialized
    if not _http_initialized:
        _http_env.reset()
        _http_initialized = True
    s = _http_env.state()
    return JSONResponse(content=s.model_dump(), status_code=200)


@app.post("/grade")
async def grade() -> JSONResponse:
    """Run the grader for the current task and return the score."""
    global _http_initialized
    if not _http_initialized:
        return JSONResponse(
            content={"error": "No episode started — call /reset first"},
            status_code=400,
        )
    result = _http_env.grade_episode()
    return JSONResponse(content=result.model_dump(), status_code=200)



# WebSocket endpoint (OpenEnv EnvClient protocol)


class WebSocketSession:
    """Per-connection state container."""
    def __init__(self) -> None:
        self.env = TradingEnvironment()
        self.session_id = str(uuid.uuid4())
        self.initialized = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket handler following the OpenEnv message protocol.

    Client → Server messages:
      {"action": "reset", "task_id"?: str, "seed"?: int}
      {"action": "state"}
      {"action": {"action_type": str, "amount": float, "limit_price"?: float}}

    Server → Client responses:
      {"observation": {...}, "reward": float|null, "done": bool, "state": {...}}
    """
    await websocket.accept()
    session = WebSocketSession()
    logger.info(f"WS session opened: {session.session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            action_data = msg.get("action")

            # ----- RESET -----
            if action_data == "reset" or msg.get("type") == "reset":
                task_id = msg.get("task_id", "easy_profitable_baseline")
                seed = msg.get("seed", None)
                obs = session.env.reset(task_id=task_id, seed=seed)
                session.initialized = True
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": None,
                    "done": False,
                    "state": session.env.state().model_dump(),
                })

            # ----- STATE -----
            elif action_data == "state" or msg.get("type") == "state":
                if not session.initialized:
                    session.env.reset()
                    session.initialized = True
                s = session.env.state()
                await websocket.send_json({"state": s.model_dump()})

            # ----- GRADE -----
            elif action_data == "grade" or msg.get("type") == "grade":
                if not session.initialized:
                    await websocket.send_json({"error": "No episode started"})
                    continue
                result = session.env.grade_episode()
                await websocket.send_json({"grade": result.model_dump()})

            # ----- STEP -----
            elif isinstance(action_data, dict):
                if not session.initialized:
                    session.env.reset()
                    session.initialized = True
                try:
                    trade_action = TradeAction(
                        action_type=action_data.get("action_type", "hold"),
                        amount=float(action_data.get("amount", 0.0)),
                        limit_price=action_data.get("limit_price"),
                    )
                    obs, reward, done, info = session.env.step(trade_action)
                    await websocket.send_json({
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "state": session.env.state().model_dump(),
                    })
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

            else:
                await websocket.send_json({"error": f"Unknown action: {action_data}"})

    except WebSocketDisconnect:
        logger.info(f"WS session closed: {session.session_id}")
    except Exception as e:
        logger.error(f"WS error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass



# Entry point
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
