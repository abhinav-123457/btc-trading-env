---
title: BTC Algorithmic Trading OpenEnv
emoji: 🪙
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - finance
  - rl
  - trading
  - bitcoin
app_port: 7860
---

# 🪙 BTC Algorithmic Trading OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

A **production-grade reinforcement learning environment** for training and evaluating
AI agents on cryptocurrency trading execution tasks. Agents act as automated Bitcoin
portfolio managers, receiving rich market observations and executing trades against a
realistic procedurally-generated market simulator.

---

## 🌍 Motivation

Algorithmic trading is one of the most high-stakes real-world domains for AI agents.
It demands:

- **Sequential decision-making** under uncertainty (partial observability)
- **Risk management** (drawdown limits, position sizing)
- **Market microstructure awareness** (bid-ask spread, slippage, fees)
- **Regime adaptation** (calm → volatile → crash → recovery)

This environment fills a gap in the OpenEnv ecosystem by providing a **finance domain**
environment with realistic mechanics, multiple difficulty levels, and a Sharpe-ratio
grader that genuinely challenges frontier models.

---

## 📐 Architecture

```
┌──────────────────────────────────────┐
│          LLM / RL Agent              │
│  (reads observation, outputs action) │
└──────────────┬───────────────────────┘
               │ HTTP POST /step  or  WebSocket /ws
               ▼
┌──────────────────────────────────────┐
│       FastAPI Server  (app.py)       │
│  POST /reset  POST /step  GET /state │
│  POST /grade  WebSocket /ws          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│    TradingEnvironment  (env.py)      │
│  ┌────────────┐  ┌────────────────┐  │
│  │  Market    │  │  Matching      │  │
│  │  Simulator │  │  Engine        │  │
│  │  (GBM +    │  │  (spread,      │  │
│  │   regimes) │  │   slippage,    │  │
│  └────────────┘  │   fees)        │  │
│                  └────────────────┘  │
│  ┌──────────────────────────────┐    │
│  │  Portfolio Tracker           │    │
│  │  (PnL, drawdown, Sharpe)     │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

---

## 🔭 Observation Space

Each call to `reset()` or `step()` returns a `TradingObservation`:

| Field | Type | Description |
|---|---|---|
| `tick.bid` | float | Best bid price (USD) |
| `tick.ask` | float | Best ask price (USD) |
| `tick.mid` | float | Mid price (USD) |
| `tick.volume` | float | Tick volume (BTC) |
| `tick.spread_pct` | float | Bid-ask spread as % of mid |
| `portfolio.fiat_balance` | float | USD cash available |
| `portfolio.btc_balance` | float | BTC held |
| `portfolio.portfolio_value` | float | Total value (USD) |
| `portfolio.unrealized_pnl_pct` | float | Unrealized PnL % |
| `portfolio.max_drawdown` | float | Max drawdown so far (0–1) |
| `indicators.sma_5/20/50` | float | Simple moving averages |
| `indicators.rsi_14` | float | RSI (0–100) |
| `indicators.macd` | float | MACD line |
| `indicators.volatility_20` | float | 20-step rolling log-return std |
| `indicators.price_history` | List[float] | Last 20 mid prices |
| `step_num` | int | Current step |
| `max_steps` | int | Episode length |
| `flash_crash_active` | bool | True during a flash crash event |
| `episode_return_pct` | float | Cumulative return % |
| `done` | bool | Episode finished flag |
| `reward` | float | Reward received (null on reset) |

---

## 🕹️ Action Space

```python
TradeAction(
    action_type: Literal["buy", "sell", "hold"],
    amount: float,          # 0.0–1.0 fraction of available balance
    limit_price: float | None = None  # None = market order
)
```

- **buy**: Spend `amount * fiat_balance` USD to buy BTC at ask (taker) or limit
- **sell**: Liquidate `amount * btc_balance` BTC at bid (taker) or limit  
- **hold**: Do nothing
- **Fees**: 0.10% taker | 0.05% maker

---

## 🎯 Tasks

### Task 1 — Easy: Profitable Baseline

> Difficulty: ⭐

End the 100-step episode with **≥ 1% return** without breaching a **30% max drawdown**.

**Grader** (score 0.0–1.0):
- 60% — return component (linear, 0→1 for 0–5% return)
- 40% — drawdown component (1 − drawdown/0.30)

**Expected baseline score**: ~0.55–0.65

---

### Task 2 — Medium: Flash Crash Survival

> Difficulty: ⭐⭐

A random **flash crash** (8–18% price drop) is injected between steps 50–105.
Survive it, maintain overall profitability, and keep max drawdown **below 20%**.

**Grader** (score 0.0–1.0):
- 33% — drawdown survival score
- 33% — profitability score  
- 33% — crash-behavior win-rate proxy

**Expected baseline score**: ~0.40–0.55

---

### Task 3 — Hard: Sharpe Ratio Optimisation

> Difficulty: ⭐⭐⭐

Over 200 steps with **dynamically widening spreads** and flash crash events,
achieve **Sharpe ratio ≥ 0.5** and **≥ 2% return**,
risk-adjusted trading.

**Grader** (score 0.0–1.0):
- 40% — Sharpe ratio (scaled: Sharpe -0.5→1.5 maps to score 0→1)
- 30% — return score (0 → 1 for 0–5%)
- 30% — trade efficiency (win-rate minus fee-waste penalty)

**Expected baseline score**: ~0.25–0.40

---

## 🏆 Reward Function

Dense reward at every step with partial progress signals:

| Component | Range | Description |
|---|---|---|
| PnL step reward | ±5.0 | Scaled incremental portfolio change |
| Fee penalty | ≤ 0.0 | Penalises paying > 0.1% of portfolio in fees per step |
| Drawdown penalty | ≤ 0.0 | Non-linear penalty beyond 10% drawdown |
| Crash survival bonus | +0.5 | Reward for defensive action during crash |
| Terminal bonus | ±20+ | Large outcome-based shaping at episode end |

---

## 🛠️ Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/YOUR_NAME/btc-trading-env
cd btc-trading-env
pip install -r requirements.txt
python app.py   # starts on http://localhost:7860
```

### Docker

```bash
docker build -t btc-trading-env .
docker run -p 7860:7860 btc-trading-env

# Verify health
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_profitable_baseline", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "buy", "amount": 0.3, "limit_price": null}'

# Get state
curl http://localhost:7860/state

# Grade the episode
curl -X POST http://localhost:7860/grade
```

### OpenEnv Python Client (async)

```python
import asyncio, json, websockets

async def main():
    uri = "ws://localhost:7860/ws"
    async with websockets.connect(uri) as ws:
        # Reset
        await ws.send(json.dumps({"action": "reset", "task_id": "easy_profitable_baseline", "seed": 42}))
        result = json.loads(await ws.recv())
        print("Initial obs:", result["observation"]["portfolio"])
        
        # Step loop
        for _ in range(100):
            await ws.send(json.dumps({"action": {"action_type": "hold", "amount": 0.0}}))
            result = json.loads(await ws.recv())
            if result["done"]:
                break
        
        # Grade
        await ws.send(json.dumps({"action": "grade"}))
        grade = json.loads(await ws.recv())
        print("Score:", grade["grade"]["score"])

asyncio.run(main())
```

### Running Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-key-here"

python inference.py
```

**Without LLM** (rule-based fallback auto-activates when `HF_TOKEN` is not set):
```bash
python inference.py
```

---

## 📊 Baseline Scores

Evaluated with `seed=42` on a 2-vCPU / 8 GB machine. Runtime < 5 minutes.
Market data: **real Binance BTCUSDT hourly prices** (5000 candles, cached locally).
```json
{
  "scores": {
    "easy_profitable_baseline": 0.6862,
    "medium_crash_survival":    0.6959,
    "hard_sharpe_optimization": 0.3027
  },
  "average": 0.5616,
  "data_source": "Binance BTCUSDT 1h (real)"
}
```

| Task | Difficulty | Rule-Based Agent | LLM Agent |
|---|---|---|---|
| `easy_profitable_baseline` | ⭐ Easy   | 0.69       | **0.86 ✓ PASS** (llama-4-scout) |
| `medium_crash_survival`    | ⭐⭐ Medium | 0.70       | **0.81 ✓ PASS** (llama-4-scout) |
| `hard_sharpe_optimization` | ⭐⭐⭐ Hard  | 0.30       | **0.32** (100/200 steps only)   |
  Average: 0.6605
| `easy_profitable_baseline` | ⭐ Easy   | 0.69       | **0.70 Fail** (llama-3.1-8b) |
| `medium_crash_survival`    | ⭐⭐ Medium | 0.70       | **0.50 Fail** (llama-3.1-8b) |
| `hard_sharpe_optimization` | ⭐⭐⭐ Hard  | 0.30       | **0.32** (100/200 steps only)   |
  Average: 0.4558

> Scores are fully reproducible: `python inference.py` with `seed=42`
> always produces identical values. The rule-based fallback activates
> automatically when `HF_TOKEN` is not set.
> On first run, real BTC prices are fetched from Binance public API
> and cached to `btc_prices.csv` — no API key required.
---

## 📁 Project Structure
```
btc-trading-env/
├── models.py          # Pydantic models: Action, Observation, State, GraderResult
├── environment.py     # Core logic: MarketSimulator, MatchingEngine, Graders
├── server/
│   └── app.py         # FastAPI server: /reset, /step, /state, /grade, /ws
├── openenv.yaml       # OpenEnv manifest
├── inference.py       # Baseline evaluation script (OpenAI client)
├── btc_prices.csv     # Cached real Binance BTCUSDT hourly prices
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition for HF Spaces
└── README.md          # This file
```

---

## 🔬 Design Decisions

**Market data**: Real BTC/USDT hourly closing prices fetched from the Binance
public REST API (no API key required) and cached locally to `btc_prices.csv`.
~5000 candles (~6 months) are downloaded on first run. If Binance is unavailable,
the environment falls back to Geometric Brownian Motion with stochastic volatility
regimes — fully deterministic given `seed`.

**Matching engine**: Simplified but realistic: taker vs maker fees, limit order
fill-or-miss logic, market orders always fill at current bid/ask.

**Flash crash**: Injected at a random step in the episode's middle third, causing
an 8–18% instantaneous price drop followed by a slow recovery. The crash step is
deterministic given the episode seed.

**Reward shaping**: Dense signal at every step prevents the sparse-reward failure
mode. The terminal component ensures the agent optimises for the actual task objective.

**Sharpe computation**: Annualised from 30-second step log-returns
(2880 steps/day × 365 = ~1M steps/year scaling factor).

---

## License

MIT
