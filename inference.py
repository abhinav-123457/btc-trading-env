
"""
inference.py — Baseline agent evaluation script for the BTC Trading OpenEnv.

Reads credentials from environment variables:
  API_BASE_URL  — LLM API base URL (OpenAI-compatible)
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face / API key (used as the Bearer token)

Runs the LLM agent against all 3 tasks and prints final grader scores.
Total runtime is well under 20 minutes on a 2-vCPU / 8 GB machine.

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o-mini"
  export HF_TOKEN="sk-..."
  python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

# ── Inline environment (no server required) ───────────────────────────────────
# We import the environment directly so the inference script can run without
# starting the HTTP server — the grader competition runner will do the same.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import TASK_CONFIGS, TradingEnvironment
from models import TradeAction

# ── OpenAI client setup ────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    sys.exit(1)

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — API calls may fail if authentication is required.")

client = OpenAI(
    api_key=HF_TOKEN or "no-key",
    base_url=API_BASE_URL,
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Bitcoin algorithmic trading agent operating in a simulation.

At each step you receive:
- Current market tick: bid, ask, mid price, volume, spread %
- Portfolio state: fiat balance (USD), BTC balance, total portfolio value, PnL, max drawdown
- Technical indicators: SMA5, SMA20, SMA50, RSI14, MACD, volatility20
- Episode progress: step number, max steps, task ID, flash_crash_active flag

Your objective depends on the task:
- easy_profitable_baseline: End with >1% return without exceeding 30% drawdown.
- medium_crash_survival: Survive a flash crash, stay profitable, keep drawdown <20%.
- hard_sharpe_optimization: Achieve Sharpe ratio >1.0 and >2% return over 200 steps.

Decision rules:
1. When RSI < 30 and SMA5 > SMA20 (oversold + momentum reversal): BUY
2. When RSI > 70 and SMA5 < SMA20 (overbought + momentum drop): SELL
3. When flash_crash_active is True: SELL to reduce exposure (amount 0.7)
4. When max_drawdown > 0.15: SELL to de-risk (amount 0.5)
5. Default: HOLD

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "action_type": "buy" | "sell" | "hold",
  "amount": <float 0.0-1.0>,
  "limit_price": <float or null>,
  "reasoning": "<brief reason>"
}"""


# ── Observation formatter ──────────────────────────────────────────────────────

def format_observation(obs_dict: Dict[str, Any]) -> str:
    """Format observation dict into a readable string for the LLM."""
    tick = obs_dict.get("tick", {})
    portfolio = obs_dict.get("portfolio", {})
    indicators = obs_dict.get("indicators", {})

    return (
        f"=== Step {obs_dict.get('step_num')}/{obs_dict.get('max_steps')} "
        f"[Task: {obs_dict.get('task_id')}] ===\n"
        f"MARKET: mid=${tick.get('mid', 0):,.2f} | bid=${tick.get('bid', 0):,.2f} | "
        f"ask=${tick.get('ask', 0):,.2f} | spread={tick.get('spread_pct', 0):.3f}% | "
        f"volume={tick.get('volume', 0):.4f} BTC\n"
        f"PORTFOLIO: fiat=${portfolio.get('fiat_balance', 0):,.2f} | "
        f"BTC={portfolio.get('btc_balance', 0):.6f} | "
        f"value=${portfolio.get('portfolio_value', 0):,.2f} | "
        f"pnl={portfolio.get('unrealized_pnl_pct', 0):+.2f}% | "
        f"drawdown={portfolio.get('max_drawdown', 0) * 100:.2f}%\n"
        f"INDICATORS: RSI={indicators.get('rsi_14', 50):.1f} | "
        f"SMA5=${indicators.get('sma_5', 0):,.2f} | "
        f"SMA20=${indicators.get('sma_20', 0):,.2f} | "
        f"MACD={indicators.get('macd', 0):.4f} | "
        f"vol20={indicators.get('volatility_20', 0):.5f}\n"
        f"CRASH_ACTIVE={obs_dict.get('flash_crash_active', False)} | "
        f"episode_return={obs_dict.get('episode_return_pct', 0):+.2f}%"
    )


# ── LLM agent ─────────────────────────────────────────────────────────────────

def llm_decide(obs_dict: Dict[str, Any], retries: int = 2) -> TradeAction:
    """Query the LLM and parse its trading decision."""
    prompt = format_observation(obs_dict)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()
            # Strip potential markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            return TradeAction(
                action_type=parsed.get("action_type", "hold"),
                amount=float(parsed.get("amount", 0.0)),
                limit_price=parsed.get("limit_price"),
            )
        except Exception as e:
            if attempt < retries:
                print(f"  [WARN] LLM parse error (attempt {attempt+1}): {e} — retrying")
                time.sleep(1)
            else:
                print(f"  [WARN] LLM failed after {retries+1} attempts: {e} — defaulting to HOLD")
                return TradeAction(action_type="hold", amount=0.0)


def rule_based_fallback(obs_dict: Dict[str, Any]) -> TradeAction:
    """
    Deterministic rule-based agent — MA crossover + RSI + multi-layer crash guard.
    Used when LLM is unavailable or API key is not set.
    """
    indicators = obs_dict.get("indicators", {})
    portfolio = obs_dict.get("portfolio", {})
    tick = obs_dict.get("tick", {})

    rsi = indicators.get("rsi_14", 50)
    sma5 = indicators.get("sma_5", 0)
    sma20 = indicators.get("sma_20", 1)
    volatility = indicators.get("volatility_20", 0)
    crash = obs_dict.get("flash_crash_active", False)
    drawdown = portfolio.get("max_drawdown", 0)
    pnl_pct = portfolio.get("unrealized_pnl_pct", 0)
    btc_bal = portfolio.get("btc_balance", 0)
    fiat_bal = portfolio.get("fiat_balance", 0)
    mid = tick.get("mid", 1)
    spread_pct = tick.get("spread_pct", 0.05)

    btc_value = btc_bal * mid

    # ── Layer 1: Hard stop — flash crash flag ─────────────────────────────
    if crash and btc_bal > 0:
        return TradeAction(action_type="sell", amount=0.85)

    # ── Layer 2: Drawdown stop-loss ───────────────────────────────────────
    if drawdown > 0.15 and btc_bal > 0:
        return TradeAction(action_type="sell", amount=0.7)
    if drawdown > 0.10 and btc_bal > 0 and rsi > 60:
        return TradeAction(action_type="sell", amount=0.5)

    # ── Layer 3: High-volatility de-risk ──────────────────────────────────
    if volatility > 0.012 and btc_bal > 0:
        return TradeAction(action_type="sell", amount=0.4)

    # ── Layer 4: Avoid wide-spread trades (hard task) ─────────────────────
    if spread_pct > 0.20:
        return TradeAction(action_type="hold", amount=0.0)

    # ── Layer 5: Trend-following MA crossover + RSI filter ────────────────
    bullish = sma5 > sma20 * 1.002 and rsi < 62
    bearish = sma5 < sma20 * 0.998 and rsi > 52

    # Only enter long if we have room (< 60% in BTC)
    total_value = fiat_bal + btc_value
    btc_alloc = btc_value / max(total_value, 1)

    if bullish and fiat_bal > 100 and btc_alloc < 0.60:
        return TradeAction(action_type="buy", amount=0.30)
    elif bearish and btc_value > 50:
        return TradeAction(action_type="sell", amount=0.45)

    # ── Layer 6: Profit-taking at strong PnL ─────────────────────────────
    if pnl_pct > 3.0 and btc_value > 50 and rsi > 65:
        return TradeAction(action_type="sell", amount=0.3)

    return TradeAction(action_type="hold", amount=0.0)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    seed: int = 42,
    use_llm: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single episode and return the grader result."""
    env = TradingEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    step = 0
    total_reward = 0.0
    done = False
    print(f"[START] task={task_id}", flush=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id}  |  max_steps={max_steps}  |  seed={seed}")
        print(f"{'='*60}")

    while not done and step < max_steps:
        # Choose agent (LLM or rule-based fallback)
        if use_llm and HF_TOKEN:
            action = llm_decide(obs_dict)
        else:
            action = rule_based_fallback(obs_dict)

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        total_reward += reward
        step += 1
        print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

        if verbose and step % 20 == 0:
            p = obs_dict.get("portfolio", {})
            print(
                f"  step={step:>3} | action={action.action_type:<4} amt={action.amount:.2f} | "
                f"value=${p.get('portfolio_value', 0):>9,.2f} | "
                f"return={p.get('unrealized_pnl_pct', 0):+.2f}% | "
                f"reward={reward:+.3f}"
            )

    # Grade the episode
    grade = env.grade_episode()
    print(f"[END] task={task_id} score={grade.score:.4f} steps={step}", flush=True)
    if verbose:
        print(f"\n  --- Episode Summary ---")
        print(f"  Steps completed : {step}")
        print(f"  Total reward    : {total_reward:.4f}")
        print(f"  Grader score    : {grade.score:.4f}  (passed={grade.passed})")
        print(f"  Details         : {json.dumps(grade.details, indent=4)}")

    return {
        "task_id": task_id,
        "score": grade.score,
        "passed": grade.passed,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "details": grade.details,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BTC Trading OpenEnv — Baseline Inference Script")
    print("=" * 60)
    print(f"API_BASE_URL : {API_BASE_URL}")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"HF_TOKEN     : {'[SET]' if HF_TOKEN else '[NOT SET] — using rule-based fallback'}")

    # Detect whether LLM is actually usable
    use_llm = bool(HF_TOKEN)
    if use_llm:
        try:
            # Quick connectivity test
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
            )
            print("LLM connectivity: OK")
        except Exception as e:
            print(f"LLM connectivity: FAILED ({e}) — falling back to rule-based agent")
            use_llm = False

    tasks = list(TASK_CONFIGS.keys())
    results = []

    for task_id in tasks:
        result = run_episode(task_id=task_id, seed=42, use_llm=use_llm, verbose=True)
        results.append(result)
        # Small sleep between tasks to avoid rate limits
        time.sleep(1)

    # ── Final scoreboard ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    difficulties = ["EASY", "MEDIUM", "HARD"]
    total = 0.0
    for i, r in enumerate(results):
        score = r["score"]
        total += score
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(
            f"  [{difficulties[i]:>6}] {r['task_id']:<35} "
            f"score={score:.4f}  {status}"
        )

    avg = total / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print("=" * 60)

    # Machine-readable output for the competition runner
    output = {
        "scores": {r["task_id"]: r["score"] for r in results},
        "average": round(avg, 4),
        "details": results,
    }
    print("\nJSON_OUTPUT:", json.dumps(output))


if __name__ == "__main__":
    main()
