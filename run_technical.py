from src.config import load_config
from src.data import join_timeframe_features, load_data
from src.features import add_aoi_features, clean_features
from src.indicators import add_indicators
from src.metrics import calculate_metrics
from src.technical.backtest import backtest
from src.technical.strategy import generate_signals


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

    confirmation_timeframes = [
        ("m1", "use_m1_confirmation", "m1_parquet", 1),
        ("m5", "use_m5_confirmation", "m5_parquet", 5),
        ("m15", "use_m15_confirmation", "lower_timeframe_parquet", 15),
        ("m30", "use_m30_confirmation", "m30_parquet", 30),
        ("h1", "use_h1_confirmation", "h1_parquet", 60),
    ]

    for timeframe, enabled_key, path_key, timeframe_minutes in confirmation_timeframes:
        if not config["strategy"].get(enabled_key, False):
            continue

        lower = load_data(config["data"][path_key])
        lower = add_indicators(lower)
        df = join_timeframe_features(
            df,
            lower,
            timeframe,
            timeframe_minutes,
            config["data"]["timeframe_minutes"],
        )
        print(df)
        print(f"Joined {lower.height:,} {timeframe.upper()} candles")

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