from src.config import load_config
from src.data import load_data
from src.indicators import add_indicators
from src.features import (
    add_aoi_features,
    clean_features
)
from src.technical.strategy import generate_signals
from src.technical.backtest import backtest
from src.metrics import calculate_metrics


def main():

    config = load_config()

    path = config["data"]["parquet"]

    print("Loading data...")

    df = load_data(path)

    print(
        f"Loaded {df.height:,} candles"
    )

    print("Calculating indicators...")

    df = add_indicators(df)

    print("Calculating AOI...")

    df = add_aoi_features(df)

    df = clean_features(df)

    print("Generating signals...")

    df = generate_signals(
        df,
        config
    )

    strategy = config["strategy"]

    print("Running backtest...")

    trades, balance = backtest(
        df,
        initial_balance=10_000,
        risk_per_trade=
            strategy["risk_per_trade"],
        sl_atr=
            strategy["atr_sl_multiplier"],
        tp_atr=
            strategy["atr_tp_multiplier"],
        cost_pips=
            strategy["transaction_cost_pips"]
    )

    metrics = calculate_metrics(
        trades,
        10_000
    )

    print("\n===== TECHNICAL STRATEGY =====")

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        f"\nFinal balance: "
        f"{balance:.2f}"
    )


if __name__ == "__main__":
    main()