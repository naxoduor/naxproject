import numpy as np


def backtest(
    df,
    initial_balance=10_000,
    risk_per_trade=0.01,
    sl_atr=1.5,
    tp_atr=3.0,
    pip_size=0.0001,
    cost_pips=0.6
):

    timestamps = df["timestamp"].to_list()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    signals = df["signal"].to_numpy()

    balance = initial_balance

    trades = []

    position = 0
    entry = 0
    stop = 0
    target = 0
    entry_time = None

    risk_amount = 0

    for i in range(1, len(df)):

        if position == 0:

            signal = signals[i - 1]

            if signal == 0:
                continue

            entry = opens[i]

            current_atr = atr[i - 1]

            if np.isnan(current_atr):
                continue

            risk_amount = balance * risk_per_trade

            cost = cost_pips * pip_size

            if signal == 1:

                position = 1

                stop = entry - (
                    current_atr * sl_atr
                )

                target = entry + (
                    current_atr * tp_atr
                )

            else:

                position = -1

                stop = entry + (
                    current_atr * sl_atr
                )

                target = entry - (
                    current_atr * tp_atr
                )

            entry_time = timestamps[i]

            continue

        # LONG
        if position == 1:

            if lows[i] <= stop:

                exit_price = stop

                pnl = (
                    exit_price - entry
                ) / pip_size

                pnl -= cost_pips

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "side": "LONG",
                    "entry": entry,
                    "exit": exit_price,
                    "pips": pnl,
                    "pnl": risk_amount *
                          (pnl /
                           ((entry - stop) /
                            pip_size))
                })

                balance += trades[-1]["pnl"]

                position = 0

            elif highs[i] >= target:

                exit_price = target

                pnl = (
                    exit_price - entry
                ) / pip_size

                pnl -= cost_pips

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "side": "LONG",
                    "entry": entry,
                    "exit": exit_price,
                    "pips": pnl,
                    "pnl": risk_amount *
                          (pnl /
                           ((entry - stop) /
                            pip_size))
                })

                balance += trades[-1]["pnl"]

                position = 0

        # SHORT
        elif position == -1:

            if highs[i] >= stop:

                exit_price = stop

                pnl = (
                    entry - exit_price
                ) / pip_size

                pnl -= cost_pips

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "side": "SHORT",
                    "entry": entry,
                    "exit": exit_price,
                    "pips": pnl,
                    "pnl": risk_amount *
                          (pnl /
                           ((stop - entry) /
                            pip_size))
                })

                balance += trades[-1]["pnl"]

                position = 0

            elif lows[i] <= target:

                exit_price = target

                pnl = (
                    entry - exit_price
                ) / pip_size

                pnl -= cost_pips

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "side": "SHORT",
                    "entry": entry,
                    "exit": exit_price,
                    "pips": pnl,
                    "pnl": risk_amount *
                          (pnl /
                           ((stop - entry) /
                            pip_size))
                })

                balance += trades[-1]["pnl"]

                position = 0

    return trades, balance