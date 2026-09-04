import numpy as np


def backtest_ml(
    results,
    initial_balance=10_000,
    risk_per_trade=0.01,
    sl_atr=1.0,
    tp_atr=2.0,
    pip_size=0.0001,
    cost_pips=0.8,
    probability_long=0.60,
    probability_short=0.40
):

    balance = initial_balance

    trades = []

    position = 0

    entry = None
    stop = None
    target = None

    risk_amount = None

    print("print total results", len(results))

    for i in range(1, len(results)):

        current = results[i]

        if position == 0:

            if (
                current["probability_long"]
                >= probability_long
            ):

                position = 1

            elif (
                current["probability_short"]
                >=
                (1 - probability_short)
            ):

                position = -1

            else:

                continue

            entry = current["close"]

            atr = current["atr"]

            risk_amount = (
                balance *
                risk_per_trade
            )

            if position == 1:

                stop = (
                    entry -
                    atr * sl_atr
                )

                target = (
                    entry +
                    atr * tp_atr
                )

            else:

                stop = (
                    entry +
                    atr * sl_atr
                )

                target = (
                    entry -
                    atr * tp_atr
                )

            continue

        # Since the ML result currently contains
        # only close prices, use close-based exits.

        price = current["close"]

        if position == 1:

            if price <= stop:

                pips = (
                    price - entry
                ) / pip_size

                pips -= cost_pips

                pnl = (
                    risk_amount *
                    pips /
                    ((entry - stop) /
                     pip_size)
                )

                trades.append({
                    "side": "LONG",
                    "entry": entry,
                    "exit": price,
                    "pips": pips,
                    "pnl": pnl
                })

                balance += pnl

                position = 0

            elif price >= target:

                pips = (
                    price - entry
                ) / pip_size

                pips -= cost_pips

                pnl = (
                    risk_amount *
                    pips /
                    ((entry - stop) /
                     pip_size)
                )

                trades.append({
                    "side": "LONG",
                    "entry": entry,
                    "exit": price,
                    "pips": pips,
                    "pnl": pnl
                })

                balance += pnl

                position = 0

        elif position == -1:

            if price >= stop:

                pips = (
                    entry - price
                ) / pip_size

                pips -= cost_pips

                pnl = (
                    risk_amount *
                    pips /
                    ((stop - entry) /
                     pip_size)
                )

                trades.append({
                    "side": "SHORT",
                    "entry": entry,
                    "exit": price,
                    "pips": pips,
                    "pnl": pnl
                })

                balance += pnl

                position = 0

            elif price <= target:

                pips = (
                    entry - price
                ) / pip_size

                pips -= cost_pips

                pnl = (
                    risk_amount *
                    pips /
                    ((stop - entry) /
                     pip_size)
                )

                trades.append({
                    "side": "SHORT",
                    "entry": entry,
                    "exit": price,
                    "pips": pips,
                    "pnl": pnl
                })

                balance += pnl

                position = 0

    return trades, balance