import sys
sys.path.insert(0, ".")
from environment import TradingEnvironment
from models import TradeAction


def ma_crossover_agent(obs):
    ind = obs.indicators
    port = obs.portfolio
    tick = obs.tick
    btc_value = port.btc_balance * tick.mid
    total = port.fiat_balance + btc_value

    if obs.flash_crash_active and port.btc_balance > 0:
        return TradeAction(action_type="sell", amount=0.85)

    if port.max_drawdown > 0.15 and port.btc_balance > 0:
        return TradeAction(action_type="sell", amount=0.7)

    bullish = ind.sma_5 > ind.sma_20 * 1.002 and ind.rsi_14 < 65
    bearish = ind.sma_5 < ind.sma_20 * 0.998 and ind.rsi_14 > 52
    btc_alloc = btc_value / max(total, 1)

    if bullish and port.fiat_balance > 100 and btc_alloc < 0.6:
        return TradeAction(action_type="buy", amount=0.3)
    elif bearish and btc_value > 50:
        return TradeAction(action_type="sell", amount=0.45)

    return TradeAction(action_type="hold", amount=0.0)


print(f"{'seed':<8} {'sharpe':>8} {'return%':>9} {'value':>12} {'trades':>7}")
print("-" * 50)

for seed in [1, 7, 13, 42, 99, 123, 200, 500]:
    env = TradingEnvironment()
    obs = env.reset(task_id="hard_sharpe_optimization", seed=seed)
    for _ in range(200):
        action = ma_crossover_agent(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
    s = env.state()
    ret = (s.current_portfolio_value - 10000) / 10000 * 100
    print(
        f"{seed:<8} {s.sharpe_ratio:>+8.3f} {ret:>+8.2f}%"
        f"  ${s.current_portfolio_value:>10,.0f}  {s.total_trades:>6}"
    )