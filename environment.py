"""
environment.py -- BTC Trading OpenEnv environment.

Implements:
  - Real BTC price data via yfinance (cached to btc_prices.csv)
  - GBM fallback if data unavailable (fully deterministic)
  - Simulated matching engine with spread & slippage
  - Portfolio tracking with drawdown / Sharpe computation
  - Reward shaping with partial progress signals
  - 3 graded tasks: easy -> medium -> hard
"""
from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
import uuid
from typing import Dict, List, Optional, Tuple
import json
import urllib.request   
import urllib.parse

from models import (
    GraderResult,
    Portfolio,
    TechIndicators,
    TickData,
    TradeAction,
    TradingObservation,
    TradingState,
)

logger = logging.getLogger(__name__)


# Constants

INITIAL_FIAT = 10_000.0
INITIAL_BTC = 0.0
TAKER_FEE = 0.001
MAKER_FEE = 0.0005
MAX_DRAWDOWN_LIMIT = 0.30
MIN_TRADE_USD = 1.0

TASK_CONFIGS: Dict[str, dict] = {
    "easy_profitable_baseline": {
        "max_steps": 100,
        "target_return_pct": 1.0,
        "flash_crash": False,
        "dynamic_spread": False,
        "target_sharpe": None,
        "description": "End episode with >1% positive return without hitting drawdown limits.",
    },
    "medium_crash_survival": {
        "max_steps": 150,
        "target_return_pct": 0.0,
        "flash_crash": True,
        "dynamic_spread": False,
        "target_sharpe": None,
        "description": "Stay profitable through a random flash crash without >20% drawdown.",
    },
    "hard_sharpe_optimization": {
        "max_steps": 200,
        "target_return_pct": 2.0,
        "flash_crash": True,
        "dynamic_spread": True,
        "target_sharpe": 0.5,
        "description": "Achieve Sharpe >0.5 and >2% return while managing dynamic spreads.",
    },
}


# Real market data loader

_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_prices.csv")
_REAL_PRICES: List[float] = []
 
# Binance API settings
_BINANCE_URL  = "https://api.binance.com/api/v3/klines"
_SYMBOL       = "BTCUSDT"
_INTERVAL     = "1h"
_LIMIT        = 1000        # max rows per Binance request
_TARGET_ROWS  = 4320        # ~6 months of hourly data
_ONE_HOUR_MS  = 3_600_000   # milliseconds
 
 
def _fetch_page(start_ms: int, end_ms: int) -> List[float]:
    """Fetch one page of klines from Binance. Returns list of close prices."""
    params = urllib.parse.urlencode({
        "symbol":    _SYMBOL,
        "interval":  _INTERVAL,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     _LIMIT,
    })
    url = f"{_BINANCE_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return [float(row[4]) for row in data]   # index 4 = close price
 
 
def _load_real_prices() -> List[float]:
    """
    Load real BTC/USDT hourly close prices from Binance.
    Fallback chain: memory cache -> disk CSV -> Binance API -> [] (GBM fallback).
    """
    global _REAL_PRICES
    if _REAL_PRICES:
        return _REAL_PRICES
 
    # 1. Disk cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                prices = [float(row[0]) for row in reader if row]
            if len(prices) >= 200:
                logger.info(f"Loaded {len(prices)} BTC prices from disk cache.")
                _REAL_PRICES = prices
                return _REAL_PRICES
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
 
    # 2. Binance REST API
    try:
        logger.info("Fetching BTC/USDT klines from Binance public API...")
        prices: List[float] = []
        end_ms = int(time.time() * 1000)
 
        while len(prices) < _TARGET_ROWS:
            start_ms = end_ms - _LIMIT * _ONE_HOUR_MS
            chunk = _fetch_page(start_ms, end_ms)
            if not chunk:
                break
            prices = chunk + prices     # prepend so oldest is first
            end_ms = start_ms - 1
            time.sleep(0.1)             # stay well within rate limits
 
        if len(prices) < 200:
            raise ValueError(f"Only got {len(prices)} rows.")
 
        # Save to disk
        with open(_CACHE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["close"])
            for p in prices:
                writer.writerow([round(p, 2)])
 
        logger.info(f"Cached {len(prices)} hourly BTC prices to {_CACHE_FILE}")
        _REAL_PRICES = prices
        return _REAL_PRICES
 
    except Exception as e:
        logger.warning(f"Binance fetch failed: {e} -- falling back to GBM simulation.")
        return []


# Market simulator (GBM with optional real-data backbone)

class MarketSimulator:
    """
    Generates BTC tick data.

    If real prices are available: samples a random window from them and
    overlays realistic bid-ask spread, volume, and regime detection.

    If not: falls back to pure GBM with stochastic volatility regimes.
    """

    BASE_PRICE = 42_000.0
    BASE_SPREAD_BPS = 5
    HIGH_SPREAD_BPS = 40

    def __init__(
        self,
        seed: int,
        n_steps: int,
        flash_crash: bool = False,
        dynamic_spread: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
        self._n_steps = n_steps
        self._flash_crash = flash_crash
        self._dynamic_spread = dynamic_spread
        self._prices: List[float] = []
        self._volumes: List[float] = []
        self._regimes: List[str] = []
        self._generate_series()

    def _randn(self) -> float:
        u1 = self._rng.random()
        u2 = self._rng.random()
        return math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)

    def _generate_series(self) -> None:
        real = _load_real_prices()
        n_total = self._n_steps + 60

        if len(real) >= n_total + 50:
            self._generate_from_real(real, n_total)
        else:
            self._generate_gbm(n_total)

    def _generate_from_real(self, real: List[float], n_total: int) -> None:
        """Sample a window from real prices, overlay spread/volume/crash."""
        max_start = len(real) - n_total - 1
        start_idx = self._rng.randint(0, max(0, max_start))
        window = real[start_idx: start_idx + n_total]

        # Inject flash crash by scaling a window down
        crash_step = -1
        if self._flash_crash:
            crash_step = self._rng.randint(n_total // 3, int(n_total * 0.7))
            crash_factor = self._rng.uniform(0.82, 0.92)
            crash_len = self._rng.randint(3, 8)
            # Rebuild window with crash injected
            window = list(window)
            ref = window[crash_step]
            for i in range(crash_step, min(crash_step + crash_len, len(window))):
                t = i - crash_step
                factor = crash_factor + (1 - crash_factor) * (t / crash_len)
                window[i] = window[i] * factor

        for i, price in enumerate(window):
            # Detect volatility regime from local price change
            if i > 0:
                pct_chg = abs(price - window[i - 1]) / max(window[i - 1], 1)
                if pct_chg > 0.015:
                    regime = "volatile"
                elif crash_step != -1 and crash_step <= i < crash_step + 10:
                    regime = "crash"
                else:
                    regime = "normal"
            else:
                regime = "normal"

            volume = self._rng.lognormvariate(2.5, 0.8) * (
                2.5 if regime in ("volatile", "crash") else 1.0
            )
            self._prices.append(round(price, 2))
            self._volumes.append(round(volume, 4))
            self._regimes.append(regime)

    def _generate_gbm(self, n_total: int) -> None:
        """Pure GBM fallback."""
        price = self.BASE_PRICE
        regime = "normal"
        regime_counter = 0
        crash_injected = False
        crash_step = (
            self._rng.randint(self._n_steps // 3, int(self._n_steps * 0.7))
            if self._flash_crash else -1
        )

        for i in range(n_total):
            regime_counter += 1
            if regime == "normal" and regime_counter > self._rng.randint(15, 35):
                regime = self._rng.choice(["volatile", "normal"])
                regime_counter = 0
            elif regime == "volatile" and regime_counter > self._rng.randint(5, 20):
                regime = "normal"
                regime_counter = 0

            if self._flash_crash and i == crash_step and not crash_injected:
                crash_injected = True
                regime = "crash"
                price *= self._rng.uniform(0.82, 0.92)
            elif regime == "crash" and regime_counter > self._rng.randint(3, 8):
                regime = "normal"
                regime_counter = 0

            mu = 0.0003 if regime == "normal" else -0.0002
            sigma = 0.003 if regime == "normal" else 0.012
            price = max(price * math.exp(mu + sigma * self._randn()), 1000.0)
            volume = self._rng.lognormvariate(2.5, 0.8) * (
                2.5 if regime in ("volatile", "crash") else 1.0
            )
            self._prices.append(round(price, 2))
            self._volumes.append(round(volume, 4))
            self._regimes.append(regime)

    def get_tick(self, step: int, start_offset: int = 50) -> Tuple[TickData, str]:
        idx = min(step + start_offset, len(self._prices) - 1)
        mid = self._prices[idx]
        regime = self._regimes[idx]
        vol = self._volumes[idx]

        if self._dynamic_spread and regime in ("volatile", "crash"):
            spread_bps = self._rng.uniform(self.HIGH_SPREAD_BPS, self.HIGH_SPREAD_BPS * 2)
        elif self._flash_crash and regime == "crash":
            spread_bps = self.HIGH_SPREAD_BPS
        else:
            spread_bps = self.BASE_SPREAD_BPS

        half_spread = mid * spread_bps / 10_000 / 2
        bid = mid - half_spread
        ask = mid + half_spread

        return TickData(
            timestamp=int(time.time()) + step * 30,
            bid=round(bid, 2),
            ask=round(ask, 2),
            mid=round(mid, 2),
            volume=round(vol, 4),
            spread_pct=round(spread_bps / 100, 4),
        ), regime

    def price_history(self, step: int, window: int = 50, start_offset: int = 50) -> List[float]:
        end = min(step + start_offset + 1, len(self._prices))
        start = max(0, end - window)
        return [round(p, 2) for p in self._prices[start:end]]



# Technical indicators


def _sma(prices: List[float], n: int) -> float:
    if len(prices) < n:
        return prices[-1] if prices else 0.0
    return sum(prices[-n:]) / n


def _rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2)


def _ema(prices: List[float], period: int) -> float:
    if not prices:
        return 0.0
    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def _macd(prices: List[float]) -> Tuple[float, float]:
    if len(prices) < 26:
        return 0.0, 0.0
    macd_line = _ema(prices, 12) - _ema(prices, 26)
    return round(macd_line, 4), round(macd_line * 0.9, 4)


def _volatility(prices: List[float], n: int = 20) -> float:
    if len(prices) < 2:
        return 0.0
    log_returns = [
        math.log(prices[i] / prices[i - 1])
        for i in range(max(1, len(prices) - n), len(prices))
        if prices[i - 1] > 0
    ]
    if len(log_returns) < 2:
        return 0.0
    mean = sum(log_returns) / len(log_returns)
    var = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    return round(math.sqrt(var), 6)


def compute_indicators(price_history: List[float]) -> TechIndicators:
    return TechIndicators(
        sma_5=round(_sma(price_history, 5), 2),
        sma_20=round(_sma(price_history, 20), 2),
        sma_50=round(_sma(price_history, 50), 2),
        rsi_14=_rsi(price_history, 14),
        macd=_macd(price_history)[0],
        macd_signal=_macd(price_history)[1],
        volatility_20=_volatility(price_history, 20),
        price_history=price_history[-20:],
    )



# Matching engine


def execute_trade(
    action: TradeAction,
    tick: TickData,
    portfolio: Portfolio,
) -> Tuple[Portfolio, float, str]:
    note = "hold"
    fee_paid = 0.0
    p = portfolio.model_copy(deep=True)

    if action.action_type == "hold" or action.amount <= 0.0:
        return p, 0.0, "hold"

    if action.action_type == "buy":
        exec_price = tick.ask
        if action.limit_price is not None:
            if action.limit_price < tick.ask:
                return p, 0.0, "limit_not_filled"
            exec_price = min(action.limit_price, tick.ask)
            fee_rate = MAKER_FEE
        else:
            fee_rate = TAKER_FEE

        fiat_to_spend = p.fiat_balance * action.amount
        if fiat_to_spend < MIN_TRADE_USD:
            return p, 0.0, "insufficient_fiat"

        fee = fiat_to_spend * fee_rate
        btc_acquired = (fiat_to_spend - fee) / exec_price
        p.fiat_balance -= fiat_to_spend
        p.btc_balance += btc_acquired
        p.total_fees_paid += fee
        p.total_trades += 1
        fee_paid = fee
        note = f"buy {btc_acquired:.6f} BTC @ {exec_price:.2f}"

    elif action.action_type == "sell":
        exec_price = tick.bid
        if action.limit_price is not None:
            if action.limit_price > tick.bid:
                return p, 0.0, "limit_not_filled"
            exec_price = max(action.limit_price, tick.bid)
            fee_rate = MAKER_FEE
        else:
            fee_rate = TAKER_FEE

        btc_to_sell = p.btc_balance * action.amount
        if btc_to_sell * exec_price < MIN_TRADE_USD:
            return p, 0.0, "insufficient_btc"

        gross_fiat = btc_to_sell * exec_price
        fee = gross_fiat * fee_rate
        p.btc_balance -= btc_to_sell
        p.fiat_balance += gross_fiat - fee
        p.total_fees_paid += fee
        p.total_trades += 1
        fee_paid = fee
        note = f"sell {btc_to_sell:.6f} BTC @ {exec_price:.2f}"

    return p, fee_paid, note



# Reward function


def compute_reward(
    prev_value: float,
    curr_value: float,
    fee_paid: float,
    max_drawdown: float,
    action_type: str,
    flash_crash_active: bool,
    task_id: str,
    done: bool,
    portfolio: Portfolio,
    target_return_pct: float,
    total_steps: int,
    current_step: int,
) -> float:
    reward = 0.0

    pnl_pct = (curr_value - prev_value) / max(prev_value, 1.0)
    reward += max(min(pnl_pct * 100, 5.0), -5.0)

    fee_fraction = fee_paid / max(curr_value, 1.0)
    if fee_fraction > 0.001:
        reward -= (fee_fraction - 0.001) * 50

    if max_drawdown > 0.10:
        reward -= (max_drawdown - 0.10) * 20

    if flash_crash_active and action_type in ("sell", "hold"):
        reward += 0.5

    if done:
        episode_return = (portfolio.portfolio_value - INITIAL_FIAT) / INITIAL_FIAT * 100
        if episode_return >= target_return_pct:
            reward += 20.0 + min(episode_return - target_return_pct, 10.0)
        elif episode_return > 0:
            reward += episode_return * 2
        else:
            reward -= abs(episode_return) * 1.5

        if task_id == "hard_sharpe_optimization" and portfolio.total_trades > 0:
            win_rate = portfolio.winning_trades / portfolio.total_trades
            reward += win_rate * 10

    return round(reward, 4)



# Core Environment

class TradingEnvironment:
    """
    BTC Algorithmic Trading Execution Environment.

    Uses real BTC-USD hourly prices from yfinance when available,
    falls back to GBM simulation otherwise.

    Implements the OpenEnv interface:
      reset(task_id, seed) -> TradingObservation
      step(action)         -> (TradingObservation, reward, done, info)
      state()              -> TradingState
    """

    def __init__(self) -> None:
        self._portfolio: Optional[Portfolio] = None
        self._market: Optional[MarketSimulator] = None
        self._step: int = 0
        self._max_steps: int = 100
        self._task_id: str = "easy_profitable_baseline"
        self._episode_id: str = str(uuid.uuid4())
        self._done: bool = False
        self._initial_value: float = INITIAL_FIAT
        self._peak_value: float = INITIAL_FIAT
        self._returns: List[float] = []
        self._prev_value: float = INITIAL_FIAT
        self._seed: int = 42
        self._start_offset: int = 50
        self._crash_regime_active: bool = False
        self._using_real_data: bool = False

    def reset(
        self,
        task_id: str = "easy_profitable_baseline",
        seed: Optional[int] = None,
    ) -> TradingObservation:
        if task_id not in TASK_CONFIGS:
            task_id = "easy_profitable_baseline"

        self._task_id = task_id
        cfg = TASK_CONFIGS[task_id]
        self._max_steps = cfg["max_steps"]
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._done = False
        self._crash_regime_active = False

        self._portfolio = Portfolio(
            fiat_balance=INITIAL_FIAT,
            btc_balance=INITIAL_BTC,
            portfolio_value=INITIAL_FIAT,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            max_drawdown=0.0,
            peak_value=INITIAL_FIAT,
            total_trades=0,
            winning_trades=0,
            total_fees_paid=0.0,
        )
        self._initial_value = INITIAL_FIAT
        self._peak_value = INITIAL_FIAT
        self._prev_value = INITIAL_FIAT
        self._returns = [INITIAL_FIAT]

        real = _load_real_prices()
        self._using_real_data = len(real) >= self._max_steps + 60

        self._market = MarketSimulator(
            seed=self._seed,
            n_steps=self._max_steps + 10,
            flash_crash=cfg["flash_crash"],
            dynamic_spread=cfg["dynamic_spread"],
        )

        return self._build_observation()

    def step(self, action: TradeAction) -> Tuple[TradingObservation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done -- call reset() before stepping.")

        tick, regime = self._market.get_tick(self._step, self._start_offset)
        self._crash_regime_active = regime == "crash"

        prev_value = self._portfolio.portfolio_value
        btc_val_before = self._portfolio.btc_balance * tick.bid

        updated_portfolio, fee_paid, trade_note = execute_trade(
            action, tick, self._portfolio
        )
        self._portfolio = updated_portfolio

        if action.action_type == "sell" and self._portfolio.total_trades > 0:
            if fee_paid < btc_val_before * 0.05:
                self._portfolio.winning_trades = min(
                    self._portfolio.winning_trades + 1,
                    self._portfolio.total_trades,
                )

        btc_value = self._portfolio.btc_balance * tick.bid
        self._portfolio.portfolio_value = round(
            self._portfolio.fiat_balance + btc_value, 2
        )
        self._portfolio.unrealized_pnl = round(
            self._portfolio.portfolio_value - self._initial_value, 2
        )
        self._portfolio.unrealized_pnl_pct = round(
            self._portfolio.unrealized_pnl / self._initial_value * 100, 4
        )

        if self._portfolio.portfolio_value > self._peak_value:
            self._peak_value = self._portfolio.portfolio_value
        self._portfolio.peak_value = self._peak_value
        drawdown = (self._peak_value - self._portfolio.portfolio_value) / self._peak_value
        self._portfolio.max_drawdown = round(max(self._portfolio.max_drawdown, drawdown), 4)

        self._returns.append(self._portfolio.portfolio_value)
        self._step += 1

        drawdown_limit = 0.20 if self._task_id == "medium_crash_survival" else MAX_DRAWDOWN_LIMIT
        done = (
            self._step >= self._max_steps
            or self._portfolio.max_drawdown >= drawdown_limit
            or self._portfolio.portfolio_value < 100.0
        )
        self._done = done

        cfg = TASK_CONFIGS[self._task_id]
        reward = compute_reward(
            prev_value=prev_value,
            curr_value=self._portfolio.portfolio_value,
            fee_paid=fee_paid,
            max_drawdown=self._portfolio.max_drawdown,
            action_type=action.action_type,
            flash_crash_active=self._crash_regime_active,
            task_id=self._task_id,
            done=done,
            portfolio=self._portfolio,
            target_return_pct=cfg["target_return_pct"],
            total_steps=self._max_steps,
            current_step=self._step,
        )

        info = {
            "trade_note": trade_note,
            "regime": regime,
            "fee_paid": fee_paid,
            "portfolio_value": self._portfolio.portfolio_value,
            "step": self._step,
            "using_real_data": self._using_real_data,
        }

        return self._build_observation(reward=reward, done=done), reward, done, info

    def state(self) -> TradingState:
        sharpe = self._compute_sharpe()
        win_rate = (
            self._portfolio.winning_trades / self._portfolio.total_trades
            if self._portfolio and self._portfolio.total_trades > 0
            else 0.0
        )
        return TradingState(
            episode_id=self._episode_id,
            step_count=self._step,
            task_id=self._task_id,
            initial_portfolio_value=self._initial_value,
            current_portfolio_value=(
                self._portfolio.portfolio_value if self._portfolio else 0.0
            ),
            max_drawdown=self._portfolio.max_drawdown if self._portfolio else 0.0,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=self._portfolio.total_trades if self._portfolio else 0,
            is_crashed=(
                self._portfolio.max_drawdown >= MAX_DRAWDOWN_LIMIT
                if self._portfolio else False
            ),
        )

    
    # Graders
    

    def grade_episode(self) -> GraderResult:
        if self._task_id == "easy_profitable_baseline":
            return self._grade_easy()
        elif self._task_id == "medium_crash_survival":
            return self._grade_medium()
        elif self._task_id == "hard_sharpe_optimization":
            return self._grade_hard()
        return GraderResult(task_id=self._task_id, score=0.0, passed=False, details={})

    def _grade_easy(self) -> GraderResult:
        p = self._portfolio
        if p is None:
            return GraderResult(task_id="easy_profitable_baseline", score=0.0, passed=False, details={})

        episode_return_pct = (p.portfolio_value - self._initial_value) / self._initial_value * 100
        breached = p.max_drawdown >= MAX_DRAWDOWN_LIMIT

        if episode_return_pct >= 1.0:
            return_score = min(1.0, 0.6 + (episode_return_pct - 1.0) / 8)
        else:
            return_score = max(0.0, episode_return_pct / 10 + 0.5)

        dd_score = 0.0 if breached else max(0.0, 1.0 - p.max_drawdown / MAX_DRAWDOWN_LIMIT)
        score = round(0.6 * return_score + 0.4 * dd_score, 4)
        passed = episode_return_pct >= 1.0 and not breached

        return GraderResult(
            task_id="easy_profitable_baseline",
            score=score,
            passed=passed,
            details={
                "episode_return_pct": round(episode_return_pct, 4),
                "max_drawdown": round(p.max_drawdown, 4),
                "breached_drawdown_limit": breached,
                "total_trades": p.total_trades,
                "using_real_data": self._using_real_data,
            },
        )

    def _grade_medium(self) -> GraderResult:
        p = self._portfolio
        if p is None:
            return GraderResult(task_id="medium_crash_survival", score=0.0, passed=False, details={})

        MAX_DD_MEDIUM = 0.20
        episode_return_pct = (p.portfolio_value - self._initial_value) / self._initial_value * 100

        if p.max_drawdown <= 0.10:
            dd_score = 1.0
        elif p.max_drawdown <= MAX_DD_MEDIUM:
            dd_score = 1.0 - (p.max_drawdown - 0.10) / 0.10
        else:
            dd_score = 0.0

        profit_score = max(0.0, min(1.0, 0.5 + episode_return_pct / 10))
        win_rate = (p.winning_trades / p.total_trades) if p.total_trades > 0 else 0.5
        crash_score = min(1.0, win_rate * 1.5)

        survived = p.max_drawdown < MAX_DD_MEDIUM
        score = round((dd_score + profit_score + crash_score) / 3, 4)
        passed = survived and episode_return_pct >= 0.0

        return GraderResult(
            task_id="medium_crash_survival",
            score=score,
            passed=passed,
            details={
                "survived_crash": survived,
                "episode_return_pct": round(episode_return_pct, 4),
                "max_drawdown": round(p.max_drawdown, 4),
                "drawdown_limit": MAX_DD_MEDIUM,
                "win_rate": round(win_rate, 4),
                "total_trades": p.total_trades,
                "using_real_data": self._using_real_data,
            },
        )

    def _grade_hard(self) -> GraderResult:
        p = self._portfolio
        if p is None:
            return GraderResult(task_id="hard_sharpe_optimization", score=0.0, passed=False, details={})

        sharpe = self._compute_sharpe()
        episode_return_pct = (p.portfolio_value - self._initial_value) / self._initial_value * 100
        win_rate = (p.winning_trades / p.total_trades) if p.total_trades > 0 else 0.0

        TARGET_SHARPE = 0.5
        TARGET_RETURN = 2.0

        sharpe_score = max(0.0, min(1.0, (sharpe + 0.5) / 2.0))
        if episode_return_pct >= TARGET_RETURN:
            return_score = min(1.0, 0.6 + (episode_return_pct - TARGET_RETURN) / 10)
        else:
            return_score = max(0.0, episode_return_pct / (TARGET_RETURN * 2))

        fee_waste = p.total_fees_paid / max(p.portfolio_value, 1.0)
        efficiency_score = min(1.0, max(0.0, win_rate - fee_waste * 10))

        score = round(0.4 * sharpe_score + 0.3 * return_score + 0.3 * efficiency_score, 4)
        passed = sharpe >= TARGET_SHARPE and episode_return_pct >= TARGET_RETURN

        return GraderResult(
            task_id="hard_sharpe_optimization",
            score=score,
            passed=passed,
            details={
                "sharpe_ratio": round(sharpe, 4),
                "target_sharpe": TARGET_SHARPE,
                "episode_return_pct": round(episode_return_pct, 4),
                "target_return_pct": TARGET_RETURN,
                "win_rate": round(win_rate, 4),
                "total_trades": p.total_trades,
                "total_fees_paid": round(p.total_fees_paid, 4),
                "using_real_data": self._using_real_data,
            },
        )

    
    # Helpers
    

    def _build_observation(
        self,
        reward: Optional[float] = None,
        done: bool = False,
    ) -> TradingObservation:
        tick, regime = self._market.get_tick(self._step, self._start_offset)
        history = self._market.price_history(self._step, window=50, start_offset=self._start_offset)
        indicators = compute_indicators(history)
        episode_return_pct = (
            (self._portfolio.portfolio_value - self._initial_value) / self._initial_value * 100
            if self._portfolio else 0.0
        )
        return TradingObservation(
            tick=tick,
            portfolio=self._portfolio,
            indicators=indicators,
            step_num=self._step,
            max_steps=self._max_steps,
            task_id=self._task_id,
            flash_crash_active=(regime == "crash"),
            episode_return_pct=round(episode_return_pct, 4),
            done=done,
            reward=reward,
        )

    def _compute_sharpe(self) -> float:
        if len(self._returns) < 3:
            return 0.0
        log_rets = [
            math.log(self._returns[i] / self._returns[i - 1])
            for i in range(1, len(self._returns))
            if self._returns[i - 1] > 0 and self._returns[i] > 0
        ]
        if len(log_rets) < 2:
            return 0.0
        mean = sum(log_rets) / len(log_rets)
        variance = sum((r - mean) ** 2 for r in log_rets) / len(log_rets)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        return round(mean / std, 4)