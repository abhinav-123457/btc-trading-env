"""
models.py — Typed Pydantic models for the BTC Trading OpenEnv environment.
Defines Action, Observation, and State following the OpenEnv spec.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field



# Base classes (mirrors openenv.core.env_server.types when installed)


class Action(BaseModel):
    """Base action model — subclass with environment-specific fields."""
    pass


class Observation(BaseModel):
    """Base observation model. done / reward are always included."""
    done: bool = Field(False, description="Whether the episode has ended")
    reward: Optional[float] = Field(None, description="Reward received this step")


class State(BaseModel):
    """Episode-level metadata."""
    episode_id: str
    step_count: int = 0
    task_id: Optional[str] = None



# StepResult envelope


class StepResult(BaseModel):
    observation: "TradingObservation"
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)



# Trading-specific models


class TradeAction(Action):
    """
    The agent's chosen trading action for the current tick.

    action_type : "buy" | "sell" | "hold"
    amount      : fraction of available balance to deploy (0.0–1.0).
                  For buy  → fraction of fiat_balance to spend.
                  For sell → fraction of btc_balance to liquidate.
    limit_price : optional limit price; None = market order.
    """
    action_type: Literal["buy", "sell", "hold"] = Field(
        ..., description="Trade direction"
    )
    amount: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of available balance to deploy (0–1)",
    )
    limit_price: Optional[float] = Field(
        None, ge=0.0, description="Limit price in USD; None = market order"
    )


class TickData(BaseModel):
    """Single market tick snapshot."""
    timestamp: int = Field(..., description="Unix epoch seconds")
    bid: float = Field(..., description="Best bid price USD")
    ask: float = Field(..., description="Best ask price USD")
    mid: float = Field(..., description="Mid price USD")
    volume: float = Field(..., description="Tick volume (BTC)")
    spread_pct: float = Field(..., description="Bid-ask spread as % of mid")


class TechIndicators(BaseModel):
    """Rolling technical indicators computed from price history."""
    sma_5: float
    sma_20: float
    sma_50: float
    rsi_14: float          # 0–100
    macd: float            # MACD line
    macd_signal: float     # Signal line
    volatility_20: float   # 20-step rolling std-dev of log returns
    price_history: List[float] = Field(
        ..., max_length=50, description="Recent mid prices (oldest→newest)"
    )


class Portfolio(BaseModel):
    """Current portfolio snapshot."""
    fiat_balance: float = Field(..., description="USD cash available")
    btc_balance: float = Field(..., description="BTC held")
    portfolio_value: float = Field(..., description="Total value in USD")
    unrealized_pnl: float = Field(..., description="Unrealized PnL in USD")
    unrealized_pnl_pct: float = Field(..., description="Unrealized PnL %")
    max_drawdown: float = Field(..., description="Max drawdown so far (%)")
    peak_value: float = Field(..., description="Highest portfolio value seen")
    total_trades: int = 0
    winning_trades: int = 0
    total_fees_paid: float = 0.0


class TradingObservation(Observation):
    """
    Full observation returned by reset() and step().

    Combines portfolio state, current tick, technical indicators,
    and episode progress into a single typed structure.
    """
    tick: TickData
    portfolio: Portfolio
    indicators: TechIndicators
    step_num: int = Field(..., description="Current step within episode")
    max_steps: int = Field(..., description="Total steps in this episode")
    task_id: str = Field("easy_profitable_baseline", description="Active task")
    flash_crash_active: bool = Field(False, description="Whether a flash crash is occurring")
    episode_return_pct: float = Field(0.0, description="Episode return so far (%)")


class TradingState(State):
    """Extended state including episode-level summary."""
    initial_portfolio_value: float = 0.0
    current_portfolio_value: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    task_id: str = "easy_profitable_baseline"
    is_crashed: bool = False  # hit margin/drawdown limit


class GraderResult(BaseModel):
    """Output of a task grader."""
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Grader score 0.0–1.0")
    passed: bool
    details: dict = Field(default_factory=dict)
