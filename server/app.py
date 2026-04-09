"""
app.py — FastAPI application wrapping the BTC Trading Environment.

HTTP endpoints:
  GET  /          → Trading Terminal UI (served from ui.html)
  POST /reset     → TradingObservation
  POST /step      → StepResult
  GET  /state     → TradingState
  GET  /health    → {"status": "ok"}
  GET  /tasks     → task list
  POST /grade     → GraderResult
  GET  /metrics   → live metrics
  WS   /ws        → streaming WebSocket
"""
from __future__ import annotations
import os
import sys

# Fix sys.path: /app/server/app.py needs to import from /app
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))   # /app/server
_APP_ROOT   = os.path.dirname(_SERVER_DIR)                  # /app
for _p in (_APP_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from environment import TASK_CONFIGS, TradingEnvironment
from models import TradeAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("btc-trading-env")

app = FastAPI(
    title="BTC Algorithmic Trading OpenEnv",
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

# ── Shared environment ────────────────────────────────────────────────────────
_http_env = TradingEnvironment()
_http_initialized = False
_stats: Dict[str, Any] = {
    "total_resets": 0,
    "total_steps": 0,
    "total_ws_sessions": 0,
    "server_start_time": time.time(),
}

# ── Load UI HTML from disk (avoids all Python string-escaping problems) ───────
_UI_HTML_PATH = os.path.join(_SERVER_DIR, "ui.html")
try:
    with open(_UI_HTML_PATH, "r", encoding="utf-8") as _f:
        _UI_HTML = _f.read()
    logger.info(f"Loaded UI from {_UI_HTML_PATH}")
except FileNotFoundError:
    _UI_HTML = "<h1>BTC Trading OpenEnv</h1><p>ui.html not found next to app.py</p>"
    logger.warning(f"ui.html not found at {_UI_HTML_PATH}")


# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_profitable_baseline"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str = "hold"
    amount: float = 0.0
    limit_price: Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Trading terminal landing page."""
    return HTMLResponse(content=_UI_HTML, status_code=200)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "environment": "btc-trading-env",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _stats["server_start_time"], 1),
    }


@app.get("/metrics")
async def metrics():
    global _http_initialized
    uptime = time.time() - _stats["server_start_time"]
    ep = None
    if _http_initialized:
        s = _http_env.state()
        ep = {
            "task_id": s.task_id,
            "step_count": s.step_count,
            "current_portfolio_value": s.current_portfolio_value,
            "max_drawdown": s.max_drawdown,
            "sharpe_ratio": s.sharpe_ratio,
            "win_rate": s.win_rate,
            "total_trades": s.total_trades,
            "is_crashed": s.is_crashed,
        }
    return {
        "uptime_seconds": round(uptime, 1),
        "total_resets": _stats["total_resets"],
        "total_steps": _stats["total_steps"],
        "total_ws_sessions": _stats["total_ws_sessions"],
        "steps_per_second": round(_stats["total_steps"] / max(uptime, 1), 2),
        "http_episode": ep,
    }


@app.get("/tasks")
async def list_tasks():
    diffs = ["easy", "medium", "hard"]
    return {
        "tasks": [
            {"id": tid, "difficulty": diffs[i], **{k: v for k, v in cfg.items()}}
            for i, (tid, cfg) in enumerate(TASK_CONFIGS.items())
        ]
    }


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> JSONResponse:
    global _http_initialized
    task_id = request.task_id if request else "easy_profitable_baseline"
    seed    = request.seed    if request else None
    if task_id not in TASK_CONFIGS:
        return JSONResponse(
            content={"error": f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}"},
            status_code=400,
        )
    obs = _http_env.reset(task_id=task_id, seed=seed)
    _http_initialized = True
    _stats["total_resets"] += 1
    return JSONResponse(content=obs.model_dump(), status_code=200)


@app.post("/step")
async def step(request: StepRequest) -> JSONResponse:
    global _http_initialized
    if not _http_initialized:
        _http_env.reset()
        _http_initialized = True
        _stats["total_resets"] += 1
    if request.action_type not in ("buy", "sell", "hold"):
        return JSONResponse(
            content={"error": f"Invalid action_type '{request.action_type}'. Must be buy/sell/hold."},
            status_code=400,
        )
    if not (0.0 <= request.amount <= 1.0):
        return JSONResponse(content={"error": "amount must be between 0.0 and 1.0"}, status_code=400)
    action = TradeAction(
        action_type=request.action_type,
        amount=request.amount,
        limit_price=request.limit_price,
    )
    try:
        obs, reward, done, info = _http_env.step(action)
    except RuntimeError as e:
        return JSONResponse(content={"error": str(e)}, status_code=409)
    _stats["total_steps"] += 1
    return JSONResponse(
        content={"observation": obs.model_dump(), "reward": reward, "done": done, "info": info},
        status_code=200,
    )


@app.get("/state")
async def state() -> JSONResponse:
    global _http_initialized
    if not _http_initialized:
        _http_env.reset()
        _http_initialized = True
        _stats["total_resets"] += 1
    return JSONResponse(content=_http_env.state().model_dump(), status_code=200)


@app.post("/grade")
async def grade() -> JSONResponse:
    global _http_initialized
    if not _http_initialized:
        return JSONResponse(
            content={"error": "No episode started — call /reset first"},
            status_code=400,
        )
    result = _http_env.grade_episode()
    return JSONResponse(content=result.model_dump(), status_code=200)


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSSession:
    def __init__(self):
        self.env        = TradingEnvironment()
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.step_count  = 0


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    sess = WSSession()
    _stats["total_ws_sessions"] += 1
    logger.info(f"WS session opened: {sess.session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            action_data = msg.get("action")

            if action_data == "reset" or msg.get("type") == "reset":
                tid = msg.get("task_id", "easy_profitable_baseline")
                if tid not in TASK_CONFIGS:
                    await websocket.send_json({"error": f"Unknown task_id '{tid}'"})
                    continue
                obs = sess.env.reset(task_id=tid, seed=msg.get("seed"))
                sess.initialized = True
                sess.step_count  = 0
                _stats["total_resets"] += 1
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": None,
                    "done": False,
                    "state": sess.env.state().model_dump(),
                    "session_id": sess.session_id,
                })

            elif action_data == "state" or msg.get("type") == "state":
                if not sess.initialized:
                    sess.env.reset()
                    sess.initialized = True
                    _stats["total_resets"] += 1
                await websocket.send_json({"state": sess.env.state().model_dump()})

            elif action_data == "grade" or msg.get("type") == "grade":
                if not sess.initialized:
                    await websocket.send_json({"error": "No episode started"})
                    continue
                await websocket.send_json({"grade": sess.env.grade_episode().model_dump()})

            elif isinstance(action_data, dict):
                if not sess.initialized:
                    sess.env.reset()
                    sess.initialized = True
                    _stats["total_resets"] += 1
                at  = action_data.get("action_type", "hold")
                amt = float(action_data.get("amount", 0.0))
                if at not in ("buy", "sell", "hold"):
                    await websocket.send_json({"error": f"Invalid action_type '{at}'"})
                    continue
                if not (0.0 <= amt <= 1.0):
                    await websocket.send_json({"error": "amount out of range"})
                    continue
                try:
                    ta = TradeAction(
                        action_type=at,
                        amount=amt,
                        limit_price=action_data.get("limit_price"),
                    )
                    obs, reward, done, info = sess.env.step(ta)
                    sess.step_count += 1
                    _stats["total_steps"] += 1
                    await websocket.send_json({
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "state": sess.env.state().model_dump(),
                    })
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action_data!r}",
                    "hint": "Valid: 'reset', 'state', 'grade', or a dict with action_type/amount",
                })

    except WebSocketDisconnect:
        logger.info(f"WS closed: {sess.session_id} after {sess.step_count} steps")
    except Exception as e:
        logger.error(f"WS error in session {sess.session_id}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False, log_level="info")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False, log_level="info")