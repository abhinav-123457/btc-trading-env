"""
app.py — FastAPI application wrapping the BTC Trading Environment.

Exposes both HTTP REST endpoints (required by the competition pre-flight check)
and a WebSocket endpoint (used by the OpenEnv EnvClient).

HTTP endpoints:
  GET  /                                        → HTML landing page
  POST /reset           body: {task_id?, seed?} → TradingObservation
  POST /step            body: TradeAction        → StepResult
  GET  /state                                   → TradingState
  GET  /health                                  → {"status": "ok"}
  GET  /tasks                                   → list of available tasks
  POST /grade           body: {}                → GraderResult
  GET  /metrics                                 → live episode metrics snapshot

WebSocket:
  WS /ws                streaming session-based interface
"""
from __future__ import annotations
import os
import sys

# ── Fix sys.path so that models.py and environment.py (at /app) are importable
# when app.py lives at /app/server/app.py
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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from environment import TASK_CONFIGS, TradingEnvironment
from models import TradeAction, TradingObservation, TradingState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("btc-trading-env")

# ── App ────────────────────────────────────────────────────────────────────────

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

# ── Shared HTTP environment ────────────────────────────────────────────────────

_http_env = TradingEnvironment()
_http_initialized = False

# ── Simple request counter for /metrics ───────────────────────────────────────
_stats: Dict[str, Any] = {
    "total_resets": 0,
    "total_steps": 0,
    "total_ws_sessions": 0,
    "server_start_time": time.time(),
}


# ── Request / response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_profitable_baseline"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str = "hold"
    amount: float = 0.0
    limit_price: Optional[float] = None


# ── Landing page ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """HTML landing page — returns HTTP 200 (required by HF pre-flight)."""
    tasks_rows = "".join(
        f"<tr><td><code>{tid}</code></td>"
        f"<td>{'⭐' * (i + 1)}</td>"
        f"<td>{cfg['max_steps']}</td>"
        f"<td>{cfg['description']}</td></tr>"
        for i, (tid, cfg) in enumerate(TASK_CONFIGS.items())
    )

    uptime_s = int(time.time() - _stats["server_start_time"])
    uptime = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m {uptime_s % 60}s"

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>BTC Trading OpenEnv</title>
  <style>
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --text: #c9d1d9; --muted: #8b949e; --accent: #f0b429;
      --blue: #58a6ff; --green: #3fb950; --red: #f85149;
      --purple: #bc8cff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
           background: var(--bg); color: var(--text);
           padding: 2rem; max-width: 960px; margin: auto; line-height: 1.6; }
    h1 { color: var(--accent); font-size: 1.8rem; margin-bottom: 0.25rem; }
    h2 { color: var(--blue); font-size: 1.1rem; border-bottom: 1px solid var(--border);
         padding-bottom: 6px; margin: 2rem 0 1rem; }
    .badge { display: inline-block; background: #238636; color: #fff;
             padding: 2px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }
    .meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem; margin-bottom: 1rem; }
    .card { background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; padding: 1rem; }
    .card .val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .card .label { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: var(--surface); color: var(--muted); text-align: left;
         padding: 8px 12px; border-bottom: 1px solid var(--border); }
    td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--surface); }
    code { background: var(--surface); padding: 2px 6px; border-radius: 4px;
           font-size: 0.88em; color: var(--blue); }
    pre { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 1rem; overflow-x: auto;
          font-size: 0.82rem; line-height: 1.5; color: var(--text); }
    .endpoint { display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.4rem; }
    .method { font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;
              min-width: 48px; text-align: center; }
    .get  { background: #1f6feb22; color: var(--blue); border: 1px solid var(--blue); }
    .post { background: #23863622; color: var(--green); border: 1px solid var(--green); }
    .ws   { background: #bc8cff22; color: var(--purple); border: 1px solid var(--purple); }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .footer { margin-top: 3rem; color: var(--muted); font-size: 0.8rem;
              border-top: 1px solid var(--border); padding-top: 1rem; }
  </style>
</head>
<body>

  <h1>🪙 BTC Algorithmic Trading OpenEnv</h1>
  <p class="meta">
    <span class="badge">✓ Running</span>&nbsp;
    OpenEnv-compatible RL environment &nbsp;·&nbsp; v1.0.0 &nbsp;·&nbsp;
    Uptime: """ + uptime + """
  </p>

  <h2>📊 Live Stats</h2>
  <div class="grid">
    <div class="card">
      <div class="val">""" + str(_stats["total_resets"]) + """</div>
      <div class="label">Episodes Reset</div>
    </div>
    <div class="card">
      <div class="val">""" + str(_stats["total_steps"]) + """</div>
      <div class="label">Total Steps Taken</div>
    </div>
    <div class="card">
      <div class="val">""" + str(_stats["total_ws_sessions"]) + """</div>
      <div class="label">WebSocket Sessions</div>
    </div>
    <div class="card">
      <div class="val">""" + str(len(TASK_CONFIGS)) + """</div>
      <div class="label">Available Tasks</div>
    </div>
  </div>

  <h2>🎯 Tasks</h2>
  <table>
    <thead>
      <tr><th>Task ID</th><th>Difficulty</th><th>Steps</th><th>Description</th></tr>
    </thead>
    <tbody>""" + tasks_rows + """</tbody>
  </table>

  <h2>🔌 API Endpoints</h2>
  <div class="endpoint"><span class="method get">GET</span> <a href="/health">/health</a> &mdash; Health probe</div>
  <div class="endpoint"><span class="method get">GET</span> <a href="/tasks">/tasks</a> &mdash; List all tasks</div>
  <div class="endpoint"><span class="method get">GET</span> <a href="/state">/state</a> &mdash; Current episode state</div>
  <div class="endpoint"><span class="method get">GET</span> <a href="/metrics">/metrics</a> &mdash; Live server metrics</div>
  <div class="endpoint"><span class="method post">POST</span> <code>/reset</code> &mdash; Start a new episode</div>
  <div class="endpoint"><span class="method post">POST</span> <code>/step</code> &mdash; Execute one action</div>
  <div class="endpoint"><span class="method post">POST</span> <code>/grade</code> &mdash; Grade current episode</div>
  <div class="endpoint"><span class="method ws">WS</span> <code>/ws</code> &mdash; Streaming WebSocket interface</div>
  <div class="endpoint"><span class="method get">GET</span> <a href="/docs">/docs</a> &mdash; Interactive Swagger UI</div>

  <h2>⚡ Quick Start</h2>
  <pre><span style="color:var(--muted)"># 1. Reset environment</span>
curl -X POST /reset \\
  -H "Content-Type: application/json" \\
  -d '{"task_id": "easy_profitable_baseline", "seed": 42}'

<span style="color:var(--muted)"># 2. Take a step</span>
curl -X POST /step \\
  -H "Content-Type: application/json" \\
  -d '{"action_type": "buy", "amount": 0.3}'

<span style="color:var(--muted)"># 3. Grade the episode</span>
curl -X POST /grade</pre>

  <h2>🔗 WebSocket Protocol</h2>
  <pre><span style="color:var(--muted)"># Reset</span>
{"action": "reset", "task_id": "easy_profitable_baseline", "seed": 42}

<span style="color:var(--muted)"># Step</span>
{"action": {"action_type": "buy", "amount": 0.3, "limit_price": null}}

<span style="color:var(--muted)"># Grade</span>
{"action": "grade"}</pre>

  <div class="footer">
    BTC Trading OpenEnv &nbsp;·&nbsp; MIT License &nbsp;·&nbsp;
    Real Binance BTCUSDT hourly data (GBM fallback when offline)
  </div>

</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health probe — must return HTTP 200."""
    return {
        "status": "ok",
        "environment": "btc-trading-env",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _stats["server_start_time"], 1),
    }


@app.get("/metrics")
async def metrics():
    """Live server metrics — useful for monitoring and debugging."""
    global _http_initialized
    uptime = time.time() - _stats["server_start_time"]
    episode_state = None
    if _http_initialized:
        s = _http_env.state()
        episode_state = {
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
        "http_episode": episode_state,
    }


@app.get("/tasks")
async def list_tasks():
    """Return metadata for all available tasks."""
    difficulties = ["easy", "medium", "hard"]
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": difficulties[i],
                **{k: v for k, v in cfg.items()},
            }
            for i, (tid, cfg) in enumerate(TASK_CONFIGS.items())
        ]
    }


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> JSONResponse:
    """Initialise a new episode. Returns the first TradingObservation as JSON."""
    global _http_initialized
    task_id = request.task_id if request else "easy_profitable_baseline"
    seed = request.seed if request else None

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
    """Execute one action and return StepResult."""
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
        return JSONResponse(
            content={"error": "amount must be between 0.0 and 1.0"},
            status_code=400,
        )

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
        _stats["total_resets"] += 1
    return JSONResponse(content=_http_env.state().model_dump(), status_code=200)


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


# ── WebSocket endpoint (OpenEnv EnvClient protocol) ───────────────────────────

class WebSocketSession:
    """Per-connection state container."""
    def __init__(self) -> None:
        self.env = TradingEnvironment()
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.step_count = 0


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket handler following the OpenEnv message protocol.

    Client → Server messages:
      {"action": "reset", "task_id"?: str, "seed"?: int}
      {"action": "state"}
      {"action": "grade"}
      {"action": {"action_type": str, "amount": float, "limit_price"?: float}}

    Server → Client responses:
      {"observation": {...}, "reward": float|null, "done": bool, "state": {...}}
    """
    await websocket.accept()
    session = WebSocketSession()
    _stats["total_ws_sessions"] += 1
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

            # ── RESET ──────────────────────────────────────────────────────────
            if action_data == "reset" or msg.get("type") == "reset":
                task_id = msg.get("task_id", "easy_profitable_baseline")
                if task_id not in TASK_CONFIGS:
                    await websocket.send_json(
                        {"error": f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}"}
                    )
                    continue
                seed = msg.get("seed", None)
                obs = session.env.reset(task_id=task_id, seed=seed)
                session.initialized = True
                session.step_count = 0
                _stats["total_resets"] += 1
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": None,
                    "done": False,
                    "state": session.env.state().model_dump(),
                    "session_id": session.session_id,
                })

            # ── STATE ──────────────────────────────────────────────────────────
            elif action_data == "state" or msg.get("type") == "state":
                if not session.initialized:
                    session.env.reset()
                    session.initialized = True
                    _stats["total_resets"] += 1
                await websocket.send_json({"state": session.env.state().model_dump()})

            # ── GRADE ──────────────────────────────────────────────────────────
            elif action_data == "grade" or msg.get("type") == "grade":
                if not session.initialized:
                    await websocket.send_json({"error": "No episode started — send reset first"})
                    continue
                result = session.env.grade_episode()
                await websocket.send_json({"grade": result.model_dump()})

            # ── STEP ───────────────────────────────────────────────────────────
            elif isinstance(action_data, dict):
                if not session.initialized:
                    session.env.reset()
                    session.initialized = True
                    _stats["total_resets"] += 1

                action_type = action_data.get("action_type", "hold")
                if action_type not in ("buy", "sell", "hold"):
                    await websocket.send_json(
                        {"error": f"Invalid action_type '{action_type}'. Must be buy/sell/hold."}
                    )
                    continue

                amount = float(action_data.get("amount", 0.0))
                if not (0.0 <= amount <= 1.0):
                    await websocket.send_json({"error": "amount must be between 0.0 and 1.0"})
                    continue

                try:
                    trade_action = TradeAction(
                        action_type=action_type,
                        amount=amount,
                        limit_price=action_data.get("limit_price"),
                    )
                    obs, reward, done, info = session.env.step(trade_action)
                    session.step_count += 1
                    _stats["total_steps"] += 1
                    await websocket.send_json({
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "state": session.env.state().model_dump(),
                    })
                except RuntimeError as e:
                    await websocket.send_json({"error": str(e)})
                except Exception as e:
                    logger.error(f"Step error in session {session.session_id}: {e}")
                    await websocket.send_json({"error": f"Internal step error: {e}"})

            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action_data!r}",
                    "hint": "Valid actions: 'reset', 'state', 'grade', or a dict with action_type/amount",
                })

    except WebSocketDisconnect:
        logger.info(f"WS session closed: {session.session_id} after {session.step_count} steps")
    except Exception as e:
        logger.error(f"WS error in session {session.session_id}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # When server/app.py is run directly, use "server.app:app" only if
    # the parent directory is on sys.path. Since we set up sys.path above,
    # it's safer to use the module path relative to /app.
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    # When executed as python server/app.py, use app directly
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
