import numpy as np


def calculate_metrics(trades, initial_balance):

    if not trades:
        return {
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "net_profit": 0,
            "return_pct": 0,
            "max_drawdown": 0
        }

    pnls = np.array([
        t["pnl"]
        for t in trades
    ])

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = (
        len(wins) /
        len(pnls)
    )

    gross_profit = wins.sum()

    gross_loss = abs(losses.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = (
        initial_balance +
        np.cumsum(pnls)
    )

    peaks = np.maximum.accumulate(equity)

    drawdowns = (
        equity - peaks
    ) / peaks

    max_drawdown = drawdowns.min()

    net_profit = pnls.sum()

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "return_pct":
            net_profit /
            initial_balance *
            100,
        "max_drawdown_pct":
            max_drawdown * 100
    }